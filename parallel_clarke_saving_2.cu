#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cfloat>  // For DBL_MAX
#include <iomanip> // For std::setprecision
#include <chrono>  // For timing
#include <omp.h>
// For DBL_MAX

// CUDA specific headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h> // Required for grid.sync()

namespace cg = cooperative_groups;

// ===================================================================================
// KERNEL-RELATED DATA STRUCTURES & DEFINITIONS
// ===================================================================================

// Use standard integer and double types for clarity
using namespace std;

//~ Define types
using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int; // let's keep as int than unsigned. -1 is init. nodes ids 0 to n-1

const node_t DEPOT = 0;
// A simple struct for 3D points/nodes.
// This must be a POD (Plain Old Data) type for easy GPU memory copies.
struct Point
{
    double x, y, demand;
};

// The output struct that the kernel will populate.
// 'volatile' is used to advise the compiler against certain optimizations
// and ensure reads/writes happen as expected in a parallel context.
struct Saving
{
    volatile int i, j;
    volatile long long value;
};

class VRP
{
public:
    size_t size;
    demand_t capacity;
    std::vector<Point> node;
    std::vector<weight_t> dist_to_depot;

    VRP() : size(0), capacity(0) {}

    void read(const std::string &filename);
    weight_t get_dist(node_t i, node_t j) const;

    size_t getSize() const
    {
        return size;
    }
    demand_t getCapacity() const
    {
        return capacity;
    }
};

void VRP::read(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    std::string line;
    while (getline(in, line) && line.find("DIMENSION") == std::string::npos)
        ;
    if (line.find(":") != std::string::npos)
        size = stoul(line.substr(line.find(":") + 1));
    while (getline(in, line) && line.find("CAPACITY") == std::string::npos)
        ;
    if (line.find(":") != std::string::npos)
        capacity = stoul(line.substr(line.find(":") + 1));
    while (getline(in, line) && line.find("NODE_COORD_SECTION") == std::string::npos)
        ;
    node.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        int id;
        in >> id >> node[i].x >> node[i].y;
    }
    while (getline(in, line) && line.find("DEMAND_SECTION") == std::string::npos)
        ;
    for (size_t i = 0; i < size; ++i)
    {
        int id;
        in >> id >> node[i].demand;
    }
    in.close();
    dist_to_depot.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        dist_to_depot[i] = get_dist(DEPOT, i);
    }
}

weight_t VRP::get_dist(node_t i, node_t j) const
{
    double dx = node[i].x - node[j].x;
    double dy = node[i].y - node[j].y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief Calculates the total travel cost for a set of routes.
 * @param vrp The VRP instance.
 * @param routes A vector of routes, where each route is a vector of customer nodes.
 * @return The total Euclidean distance for all routes, including travel from and to the depot.
 */
weight_t calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &routes)
{
    weight_t total_cost = 0.0;
    for (const auto &route : routes)
    {
        if (route.empty())
            continue;

        node_t last_node = DEPOT;
        for (node_t current_node : route)
        {
            total_cost += vrp.get_dist(last_node, current_node);
            last_node = current_node;
        }
        // Add cost to return to the depot
        total_cost += vrp.get_dist(last_node, DEPOT);
    }
    return total_cost;
}

/**
 * @brief Verifies that the solution is valid by checking capacity constraints.
 * @param vrp The VRP instance.
 * @param routes The solution routes to verify.
 * @return True if all routes respect the vehicle capacity, false otherwise.
 */
bool verify_sol(const VRP &vrp, const std::vector<std::vector<node_t>> &routes)
{
    for (const auto &route : routes)
    {
        demand_t route_demand = 0;
        for (node_t customer : route)
        {
            if (customer < 0 || customer >= vrp.size)
                return false; // Invalid node
            route_demand += vrp.node[customer].demand;
        }
        if (route_demand > vrp.getCapacity())
        {
            return false; // Capacity violated
        }
    }
    return true;
}

// ===================================================================================
// CUDA ERROR CHECKING UTILITY (Essential for debugging)
// ===================================================================================
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
                  << " \"" << cudaGetErrorString(result) << "\" for " << func << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

// ===================================================================================
// DEVICE (GPU) UTILITY FUNCTION
// ===================================================================================

// __device__ function can only be called from other GPU functions (e.g., the kernel)
__device__ double device_euclidean_dist(const Point &a, const Point &b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// ===================================================================================
// THE COMPLETE CUDA KERNEL
// ===================================================================================

/**
 * @brief CUDA kernel to find the single best saving among all valid customer pairs.
 *
 * This kernel launches a 2D grid where each thread handles one pair (i, j).
 * It performs validity checks (route, capacity, endpoints) and calculates the saving.
 * A robust atomic reduction using a compare-and-swap loop finds the global maximum
 * saving across all threads and updates the single output struct.
 *
 * @param nodes              Device pointer to array of all nodes (depot at index 0).
 * @param customer_route_map Device pointer mapping customer ID to its route ID.
 * @param route_demands      Device pointer to an array of current demands for each route.
 * @param route_endpoints    Device pointer storing the front and back customer for each route.
 * @param best_saving_out    Device pointer to a single Saving struct to store the result.
 * @param num_customers      The total number of customers (problem size 'n').
 * @param capacity           The vehicle capacity.
 */
__global__ void find_best_saving_kernel(
    const Point *nodes,
    const node_t *customer_route_map,
    const demand_t *route_demands,
    const node_t *route_endpoints,
    Saving *best_saving_out,
    int num_customers,
    demand_t capacity)
{
    // --- 1. Map thread to a unique customer pair (i, j) ---
    // Customer IDs are 1-based, so we add 1 to the thread's calculated index.
    cg::grid_group grid = cg::this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    long long saving_value = 0;

    // Exit if thread is outside the problem space or in the lower triangle of the pair matrix.
    if (i >= (num_customers + 1) || j >= (num_customers + 1) || i >= j)
    {
    }
    else
    {
        node_t route_id_i = customer_route_map[i];
        node_t route_id_j = customer_route_map[j];
        if ((route_id_i == route_id_j || route_id_i < 0 || route_id_j < 0) || (route_demands[route_id_i] + route_demands[route_id_j] > capacity))
        {
        }
        else
        {
            node_t front_i = route_endpoints[route_id_i * 2];
            node_t back_i = route_endpoints[route_id_i * 2 + 1];
            node_t front_j = route_endpoints[route_id_j * 2];
            node_t back_j = route_endpoints[route_id_j * 2 + 1];
            if (!((i == front_i || i == back_i) && (j == front_j || j == back_j)))
            {
            }
            else
            {
                saving_value = round(device_euclidean_dist(nodes[i], nodes[0])     // dist(i, depot)
                                     + device_euclidean_dist(nodes[j], nodes[0])   // dist(j, depot)
                                     - device_euclidean_dist(nodes[i], nodes[j])); // dist(i, j)
                unsigned long long int *global_max_addr_ull = (unsigned long long int *)&(best_saving_out->value);

                atomicMax(global_max_addr_ull, saving_value);
            }
        }
    }

    grid.sync();
    if (saving_value == best_saving_out->value)
    {

        atomicMin((unsigned int *)&(best_saving_out->i), (unsigned int)i);
    }
    grid.sync();
    if (best_saving_out->i == (unsigned int)i && saving_value == best_saving_out->value)
    {
        best_saving_out->j = (unsigned int)j;
    }
}

std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp)
{
    int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    demand_t CAPACITY = vrp.getCapacity();

    std::vector<node_t> h_customer_route_map(NUM_CUSTOMERS + 1);
    std::vector<demand_t> h_route_demands(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_endpoints((NUM_CUSTOMERS + 1) * 2);

    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
    {
        // Initially, each customer is in their own route
        h_customer_route_map[i] = i;
        h_route_demands[i] = vrp.node[i].demand;
        // The start and end of the route is just the customer itself
        h_route_endpoints[i * 2] = i;     // Front
        h_route_endpoints[i * 2 + 1] = i; // Back
    }

    // --- 2. DEVICE: Allocate GPU memory ---
    Point *d_nodes;
    node_t *d_customer_route_map;
    demand_t *d_route_demands;
    node_t *d_route_endpoints;
    Saving *d_best_saving_out;

    checkCudaErrors(cudaMalloc(&d_nodes, (NUM_CUSTOMERS + 1) * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_endpoints, (NUM_CUSTOMERS + 1) * 2 * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_out, sizeof(Saving)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_nodes, vrp.node.data(), (NUM_CUSTOMERS + 1) * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_endpoints, h_route_endpoints.data(), (NUM_CUSTOMERS + 1) * 2 * sizeof(node_t), cudaMemcpyHostToDevice));

    // Initialize the output struct on the GPU to a known "worst" state
    Saving h_best_saving_init = {-1, -1, 0};
    checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));

    // --- 4. KERNEL LAUNCH ---
    dim3 threadsPerBlock(128, 128);
    dim3 numBlocks(
        (NUM_CUSTOMERS + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NUM_CUSTOMERS + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    void *args[] = {
        &d_nodes,
        &d_customer_route_map,
        &d_route_demands,
        &d_route_endpoints,
        &d_best_saving_out,
        (void *)&NUM_CUSTOMERS, // Must cast pointers to void*
        &CAPACITY};

    // --- 3. Initialize Routes (Sequential) ---
    std::vector<std::vector<node_t>> routes(vrp.size);
    for (node_t i = 1; i < vrp.size; ++i)
    {
        routes[i] = {i};
    }
    Saving h_result;

    // --- 4. Merge Routes Greedily (Sequential) ---
    int id = 0;
    while (true)
    {
        // std::cout << id++ << "\n";
        if (id > NUM_CUSTOMERS)
            break;
        id++;
        // std::cout << id << "\n";
        checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_route_endpoints, h_route_endpoints.data(), (NUM_CUSTOMERS + 1) * 2 * sizeof(node_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)find_best_saving_kernel,
            numBlocks,
            threadsPerBlock,
            args));
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(&h_result, d_best_saving_out, sizeof(Saving), cudaMemcpyDeviceToHost));
        if (h_result.value <= 1e-6)
        {
            std::cout << "No more positive savings found. Halting." << std::endl;
            std::cout << id << "\n";
            break; // Exit the while loop
        }

        node_t i = h_result.i;
        node_t j = h_result.j;
        // std::cout << i << " " << j << "\n"

        node_t route_id_i = h_customer_route_map[i];
        node_t route_id_j = h_customer_route_map[j];

        // Check if the merge is valid (different routes and combined demand is within capacity)
        if (route_id_i != route_id_j && h_route_demands[route_id_i] + h_route_demands[route_id_j] <= vrp.capacity)
        {
            // Determine the position of the merge nodes (i and j) in their respective routes
            bool i_is_front = (routes[route_id_i].front() == i);
            bool i_is_back = (routes[route_id_i].back() == i);
            bool j_is_front = (routes[route_id_j].front() == j);
            bool j_is_back = (routes[route_id_j].back() == j);

            node_t merged_into_route = -1;
            node_t consumed_route = -1;

            // Case 1: Merge End of route I -> Start of route J
            if (i_is_back && j_is_front)
            {
                routes[route_id_i].insert(routes[route_id_i].end(), routes[route_id_j].begin(), routes[route_id_j].end());
                merged_into_route = route_id_i;
                consumed_route = route_id_j;
            }
            // Case 2: Merge End of route J -> Start of route I
            else if (j_is_back && i_is_front)
            {
                routes[route_id_j].insert(routes[route_id_j].end(), routes[route_id_i].begin(), routes[route_id_i].end());
                merged_into_route = route_id_j;
                consumed_route = route_id_i;
            }
            // Case 3: Merge Start of route I -> Start of route J (requires reversing route I)
            else if (i_is_front && j_is_front)
            {
                std::reverse(routes[route_id_i].begin(), routes[route_id_i].end());
                routes[route_id_i].insert(routes[route_id_i].end(), routes[route_id_j].begin(), routes[route_id_j].end());
                merged_into_route = route_id_i;
                consumed_route = route_id_j;
            }
            // Case 4: Merge End of route I -> End of route J (requires reversing route J)
            else if (i_is_back && j_is_back)
            {
                std::reverse(routes[route_id_j].begin(), routes[route_id_j].end());
                routes[route_id_i].insert(routes[route_id_i].end(), routes[route_id_j].begin(), routes[route_id_j].end());
                merged_into_route = route_id_i;
                consumed_route = route_id_j;
            }

            // If a merge happened, update all related data structures
            if (merged_into_route != -1 && consumed_route != -1)
            {
                // 1. Update total demand of the merged route
                h_route_demands[merged_into_route] += h_route_demands[consumed_route];
                h_route_demands[consumed_route] = 0;

                // 2. Update the customer-to-route map for all nodes in the consumed route
                for (node_t customer : routes[consumed_route])
                {
                    h_customer_route_map[customer] = merged_into_route;
                }

                // 3. Update the endpoints of the newly formed merged route
                h_route_endpoints[merged_into_route * 2] = routes[merged_into_route].front();
                h_route_endpoints[merged_into_route * 2 + 1] = routes[merged_into_route].back();

                // 4. Invalidate the endpoints of the consumed route so the kernel ignores it
                h_route_endpoints[consumed_route * 2] = -1;
                h_route_endpoints[consumed_route * 2 + 1] = -1;

                // 5. Clear the vector of the consumed route
                routes[consumed_route].clear();
            }
        }
    }

    // --- 5. Finalize Routes ---
    std::vector<std::vector<node_t>> final_routes;
    for (const auto &route : routes)
    {
        if (!route.empty())
        {
            final_routes.push_back(route);
        }
    }

    checkCudaErrors(cudaFree(d_nodes));
    checkCudaErrors(cudaFree(d_customer_route_map));
    checkCudaErrors(cudaFree(d_route_demands));
    checkCudaErrors(cudaFree(d_route_endpoints));
    checkCudaErrors(cudaFree(d_best_saving_out));
    checkCudaErrors(cudaDeviceReset());

    return final_routes;
}

void tsp_approx(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, node_t ncities)
{
    node_t i, j;
    node_t ClosePt = 0;
    weight_t CloseDist;

    for (i = 1; i < ncities; i++)
        tour[i] = cities[i - 1];

    tour[0] = cities[ncities - 1];

    for (i = 1; i < ncities; i++)
    {
        weight_t ThisX = vrp.node[tour[i - 1]].x;
        weight_t ThisY = vrp.node[tour[i - 1]].y;
        CloseDist = DBL_MAX;
        for (j = ncities - 1;; j--)
        {
            weight_t ThisDist = (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
            if (ThisDist <= CloseDist)
            {
                ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
                if (ThisDist <= CloseDist)
                {
                    if (j < i)
                        break;
                    CloseDist = ThisDist;
                    ClosePt = j;
                }
            }
        }
        unsigned temp = tour[i];
        tour[i] = tour[ClosePt];
        tour[ClosePt] = temp;
    }
}

std::vector<std::vector<node_t>>
postprocess_tsp_approx(const VRP &vrp, std::vector<std::vector<node_t>> &solRoutes)
{
    std::vector<std::vector<node_t>> modifiedRoutes;

    unsigned nroutes = solRoutes.size();
    for (unsigned i = 0; i < nroutes; ++i)
    {
        unsigned sz = solRoutes[i].size();
        std::vector<node_t> cities(sz + 1);
        std::vector<node_t> tour(sz + 1);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = solRoutes[i][j];

        cities[sz] = 0;

        tsp_approx(vrp, cities, tour, sz + 1);

        vector<node_t> curr_route;
        for (unsigned kk = 1; kk < sz + 1; ++kk)
        {
            curr_route.push_back(tour[kk]);
        }

        modifiedRoutes.push_back(curr_route);
    }
    return modifiedRoutes;
}

void tsp_2opt(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, unsigned ncities)
{
    unsigned improve = 0;

    while (improve < 2)
    {
        double best_distance = 0.0;

        best_distance += vrp.get_dist(DEPOT, cities[0]);

        for (unsigned jj = 1; jj < ncities; ++jj)
        {
            best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
        }

        best_distance += vrp.get_dist(DEPOT, cities[ncities - 1]);

        for (unsigned i = 0; i < ncities - 1; i++)
        {
            for (unsigned k = i + 1; k < ncities; k++)
            {
                double new_distance = best_distance;
                if (i == 0)
                    new_distance -= vrp.get_dist(DEPOT, cities[i]);
                else
                    new_distance -= vrp.get_dist(cities[i - 1], cities[i]);

                if (k == ncities - 1)
                    new_distance -= vrp.get_dist(cities[k], DEPOT);
                else
                    new_distance -= vrp.get_dist(cities[k], cities[k + 1]);

                if (i == 0)
                    new_distance += vrp.get_dist(DEPOT, cities[k]);
                else
                    new_distance += vrp.get_dist(cities[i - 1], cities[k]);

                if (k == ncities - 1)
                    new_distance += vrp.get_dist(cities[i], DEPOT);
                else
                    new_distance += vrp.get_dist(cities[i], cities[k + 1]);

                if (new_distance < best_distance)
                {
                    improve = 0;
                    int left_ptr = i, right_ptr = k;
                    while (left_ptr <= right_ptr)
                    {
                        swap(cities[left_ptr++], cities[right_ptr--]);
                    }
                    best_distance = new_distance;
                }
            }
        }
        improve++;
    }
}
std::vector<std::vector<node_t>>
postprocess_2OPT(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes)
{
    std::vector<std::vector<node_t>> postprocessed_final_routes;

    unsigned nroutes = final_routes.size();
    for (unsigned i = 0; i < nroutes; ++i)
    {
        unsigned sz = final_routes[i].size();
        std::vector<node_t> cities(sz);
        std::vector<node_t> tour(sz);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = final_routes[i][j];

        vector<node_t> curr_route;

        if (sz > 2)
            tsp_2opt(vrp, cities, tour, sz);

        for (unsigned kk = 0; kk < sz; ++kk)
        {
            curr_route.push_back(cities[kk]);
        }

        postprocessed_final_routes.push_back(curr_route);
    }
    return postprocessed_final_routes;
}

weight_t get_total_cost_of_routes(const VRP &vrp, vector<vector<node_t>> &final_routes)
{
    weight_t total_cost = 0.0;
    for (unsigned ii = 0; ii < final_routes.size(); ++ii)
    {
        weight_t curr_route_cost = 0;
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
        for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
        {
            curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
        }
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);

        total_cost += curr_route_cost;
    }

    return total_cost;
}

std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t &minCost)
{
    std::vector<std::vector<node_t>> postprocessed_final_routes;

    auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
    auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
    auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

#pragma omp parallel for
    for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz)
    {
        vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
        vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];

        unsigned sz2 = postprocessed_route2.size();
        unsigned sz3 = postprocessed_route3.size();

        weight_t postprocessed_route2_cost = 0.0;
        postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);
        for (unsigned jj = 1; jj < sz2; ++jj)
        {
            postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
        }
        postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

        weight_t postprocessed_route3_cost = 0.0;
        postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
        for (unsigned jj = 1; jj < sz3; ++jj)
        {
            postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
        }
        postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

        if (postprocessed_route3_cost > postprocessed_route2_cost)
        {
            postprocessed_final_routes.push_back(postprocessed_route2);
        }
        else
        {
            postprocessed_final_routes.push_back(postprocessed_route3);
        }
    }

    auto postprocessed_final_routes_cost = get_total_cost_of_routes(vrp, postprocessed_final_routes);

    minCost = postprocessed_final_routes_cost;

    return postprocessed_final_routes;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename.vrp> [num_threads]" << std::endl;
        return 1;
    }

    VRP vrp;
    vrp.read(argv[1]);

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<node_t>> routes = parallel_savings_algorithm(vrp);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    weight_t total_cost = calCost(vrp, routes);
    auto postRoutes = postProcessIt(vrp, routes, total_cost);
    total_cost = calCost(vrp, postRoutes);
    bool is_valid = verify_sol(vrp, postRoutes);

    std::cout << "--- Parallel Clarke & Wright Savings Algorithm ---" << std::endl;
    std::cout << "Problem File: " << argv[1] << std::endl;
    // std::cout << "Threads Used: " << omp_get_max_threads() << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Solution Cost: " << total_cost << std::endl;
    std::cout << " Total Routes Used:  " << postRoutes.size() << std::endl;
    std::cout << "Total Time Taken:    " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID") << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}
