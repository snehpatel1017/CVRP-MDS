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

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

const node_t DEPOT = 0;

struct Point
{
    double x, y, demand;
};

struct Saving
{
    int i, j;
    long long value;
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

__device__ double device_euclidean_dist(const Point &a, const Point &b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__global__ void find_best_saving_kernel(
    const Point *nodes,
    const node_t *customer_route_map,
    const demand_t *route_demands,
    const node_t *route_endpoints,
    const weight_t *dist_to_depot,
    Saving *best_saving_out,
    int num_customers,
    demand_t capacity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Bounds check early — don't touch nodes[] if out of range.
    if (i >= (num_customers + 1) || j >= (num_customers + 1) || i >= j)
        return;

    node_t route_id_i = customer_route_map[i];
    node_t route_id_j = customer_route_map[j];
    if ((route_id_i == route_id_j || route_id_i < 0 || route_id_j < 0) ||
        (route_demands[route_id_i] + route_demands[route_id_j] > capacity))
        return;

    node_t front_i = route_endpoints[route_id_i * 2];
    node_t back_i = route_endpoints[route_id_i * 2 + 1];
    node_t front_j = route_endpoints[route_id_j * 2];
    node_t back_j = route_endpoints[route_id_j * 2 + 1];
    if (!((i == front_i || i == back_i) && (j == front_j || j == back_j)))
        return;

    long long saving_value = round(dist_to_depot[i] + dist_to_depot[j] - device_euclidean_dist(nodes[i], nodes[j]));
    if (saving_value <= 0)
        return;

    long long new_bits = (long long)(saving_value);
    long long old_bits = atomicMax(&best_saving_out->value, new_bits);

    if (old_bits < new_bits)
    {

        atomicExch((unsigned int *)&best_saving_out->i, (unsigned int)i);
        atomicExch((unsigned int *)&best_saving_out->j, (unsigned int)j);
    }
    else if (old_bits == new_bits)
    {

        atomicMin((unsigned int *)&best_saving_out->i, (unsigned int)i);

        unsigned int chosen_i = best_saving_out->i;
        if (chosen_i == (unsigned int)i)
        {
            atomicExch((unsigned int *)&best_saving_out->j, (unsigned int)j);
        }
    }
}

std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp)
{
    const int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    const demand_t CAPACITY = vrp.getCapacity();

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
    weight_t *d_dist_to_depot;

    checkCudaErrors(cudaMalloc(&d_nodes, (NUM_CUSTOMERS + 1) * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_endpoints, (NUM_CUSTOMERS + 1) * 2 * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_out, sizeof(Saving)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_nodes, vrp.node.data(), (NUM_CUSTOMERS + 1) * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_endpoints, h_route_endpoints.data(), (NUM_CUSTOMERS + 1) * 2 * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));

    // Initialize the output struct on the GPU to a known "worst" state
    Saving h_best_saving_init = {-1, -1, 0};
    checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));

    // --- 4. KERNEL LAUNCH ---
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (NUM_CUSTOMERS + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NUM_CUSTOMERS + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
        {
            std::cout << id << "\n";
            break; // Exit the while loop
        }
        id++;
        checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_route_endpoints, h_route_endpoints.data(), (NUM_CUSTOMERS + 1) * 2 * sizeof(node_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));
        find_best_saving_kernel<<<numBlocks, threadsPerBlock>>>(
            d_nodes, d_customer_route_map, d_route_demands, d_route_endpoints, d_dist_to_depot,
            d_best_saving_out, NUM_CUSTOMERS, CAPACITY);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(&h_result, d_best_saving_out, sizeof(Saving), cudaMemcpyDeviceToHost));
        if (h_result.value == 0)
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
    checkCudaErrors(cudaFree(d_dist_to_depot));
    checkCudaErrors(cudaDeviceReset());

    return final_routes;
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
    std::cout << "done parallel_savings_algorithm\n";
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    weight_t total_cost = calCost(vrp, routes);
    std::cout << "done calCOST\n";
    // auto postRoutes = postProcessIt(vrp, routes, total_cost);
    // std::cout << "done postProcessIt\n";
    // total_cost = calCost(vrp, routes);
    bool is_valid = verify_sol(vrp, routes);

    std::cout << "--- Parallel Clarke & Wright Savings Algorithm ---" << std::endl;
    std::cout << "Problem File: " << argv[1] << std::endl;
    // std::cout << "Threads Used: " << omp_get_max_threads() << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Solution Cost: " << total_cost << std::endl;
    std::cout << "Number of Routes:   " << routes.size() << std::endl;
    std::cout << "Total Time Taken:    " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID") << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}
