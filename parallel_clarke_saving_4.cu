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

using namespace std;

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
    volatile int i, j;
    volatile double value;
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
    const node_t *route_head,
    const node_t *route_tail,
    const weight_t *dist_to_depot,
    Saving *best_saving_out,
    int num_customers,
    demand_t capacity)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= (num_customers + 1) || j >= (num_customers + 1) || i >= j)
    {
        return;
    }

    // --- 2. Check validity of merging pair (i, j) ---
    node_t route_id_i = customer_route_map[i];
    node_t route_id_j = customer_route_map[j];

    if (route_id_i == route_id_j || route_id_i < 0 || route_id_j < 0)
        return;
    if (route_demands[route_id_i] + route_demands[route_id_j] > capacity)
        return;

    node_t front_i = route_head[route_id_i];
    node_t back_i = route_tail[route_id_i];
    node_t front_j = route_head[route_id_j];
    node_t back_j = route_tail[route_id_j];

    if (!((i == front_i || i == back_i) && (j == front_j || j == back_j)))
    {
        return;
    }

    // --- 3. Calculate saving if the merge is valid ---
    weight_t saving_value = dist_to_depot[i]                             // dist(i, depot)
                            + dist_to_depot[j]                           // dist(j, depot)
                            - device_euclidean_dist(nodes[i], nodes[j]); // dist(i, j)

    unsigned long long int *address_as_ull = (unsigned long long int *)&(best_saving_out->value);

    // Read the current maximum value from global memory.
    double current_max_val = __longlong_as_double(*address_as_ull);

    // This loop continues as long as this thread's saving is better than the global max.
    while (saving_value > current_max_val)
    {
        // Convert our local values to their bit representations for the atomic operation.
        unsigned long long int assumed_ull = __double_as_longlong(current_max_val);
        unsigned long long int new_val_ull = __double_as_longlong(saving_value);

        unsigned long long int prev_val_ull = atomicCAS(address_as_ull, assumed_ull, new_val_ull);

        if (prev_val_ull == assumed_ull)
        {
            // Now this thread has the exclusive right to update the i and j indices.
            best_saving_out->i = i;
            best_saving_out->j = j;
            break; // Success, exit the loop.
        }

        current_max_val = __longlong_as_double(prev_val_ull);
    }
}

__global__ void update_gpu_mempory(
    int type,
    node_t *customer_route_map,
    demand_t *route_demands,
    node_t *route_head,
    node_t *route_tail,
    node_t *next_customer,
    node_t *prev_customer,
    node_t *temporary,
    int num_customers,
    node_t i,
    node_t j,
    node_t route_id_i,
    node_t route_id_j,
    node_t head_i,
    node_t tail_i,
    node_t head_j,
    node_t tail_j)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    if (tid == 0)
    {
        if (type == 1)
        {
            next_customer[i] = j;
            prev_customer[j] = i;
            route_tail[route_id_i] = tail_j;
        }
        else if (type == 2)
        {
            next_customer[j] = i;
            prev_customer[i] = j;
            route_head[route_id_i] = head_j;
        }
        else if (type == 3)
        {
            next_customer[j] = prev_customer[j];
            next_customer[i] = j;
            prev_customer[j] = i;
            route_tail[route_id_i] = head_j;
            customer_route_map[j] = route_id_i;
        }
        else
        {
            prev_customer[i] = next_customer[i];
            next_customer[j] = i;
            prev_customer[i] = j;
            route_head[route_id_i] = tail_j;
        }

        route_demands[route_id_i] += route_demands[route_id_j];
    }
    grid.sync();

    if (type == 3)
    {
        for (node_t curr = tid; curr < num_customers; curr += total_threads)
        {
            if (customer_route_map[curr] == route_id_j)
            {
                temporary[curr] = prev_customer[curr];
            }
        }
        grid.sync();

        for (node_t curr = tid; curr < num_customers; curr += total_threads)
        {
            if (customer_route_map[curr] == route_id_j)
            {
                prev_customer[curr] = next_customer[curr];
                next_customer[curr] = temporary[curr];
            }
        }
    }
    else if (type == 4)
    {
        for (node_t curr = tid; curr < num_customers; curr += total_threads)
        {
            if (curr == i)
                continue;
            if (customer_route_map[curr] == route_id_i)
            {
                temporary[curr] = prev_customer[curr];
            }
        }
        grid.sync();
        for (node_t curr = tid; curr < num_customers; curr += total_threads)
        {
            if (curr == i)
                continue;
            if (customer_route_map[curr] == route_id_i)
            {
                prev_customer[curr] = next_customer[curr];
                next_customer[curr] = temporary[curr];
                customer_route_map[curr] = route_id_i;
            }
        }
    }
    grid.sync();
    for (node_t curr = tid; curr < num_customers; curr += total_threads)
    {
        if (customer_route_map[curr] == route_id_j)
        {
            customer_route_map[curr] = route_id_i;
        }
    }
    if (tid == 0)
    {
        route_head[route_id_j] = DEPOT;
    }
}

std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp)
{
    const int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    const demand_t CAPACITY = vrp.getCapacity();

    std::vector<node_t> h_customer_route_map(NUM_CUSTOMERS + 1);
    std::vector<demand_t> h_route_demands(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_head(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_tail(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_next_customer(vrp.size, DEPOT);
    std::vector<node_t> h_prev_customer(vrp.size, DEPOT);

    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
    {
        // Initially, each customer is in their own route
        h_customer_route_map[i] = i;
        h_route_demands[i] = vrp.node[i].demand;
        // The start and end of the route is just the customer itself
        h_route_head[i] = i;
        h_route_tail[i] = i;
    }

    // --- 2. DEVICE: Allocate GPU memory ---
    Point *d_nodes;
    node_t *d_customer_route_map;
    demand_t *d_route_demands;
    node_t *d_route_head;
    node_t *d_route_tail;
    node_t *d_next_customer;
    node_t *d_prev_customer;
    node_t *d_temporary;
    Saving *d_best_saving_out;
    weight_t *d_dist_to_depot;

    checkCudaErrors(cudaMalloc(&d_nodes, (NUM_CUSTOMERS + 1) * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_next_customer, vrp.size * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_prev_customer, vrp.size * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_temporary, vrp.size * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_out, sizeof(Saving)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_nodes, vrp.node.data(), (NUM_CUSTOMERS + 1) * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_head, h_route_head.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_tail, h_route_tail.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_next_customer, h_next_customer.data(), vrp.size * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_prev_customer, h_prev_customer.data(), vrp.size * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    // Initialize the output struct on the GPU to a known "worst" state
    Saving h_best_saving_init = {-1, -1, -DBL_MAX};
    checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));

    // --- 4. KERNEL LAUNCH ---
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
        (NUM_CUSTOMERS + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NUM_CUSTOMERS + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int threads_per_block = std::min(1024, NUM_CUSTOMERS);
    int numBlocks1D = std::min(30, (NUM_CUSTOMERS + threads_per_block - 1) / threads_per_block);

    Saving h_result;

    // --- 4. Merge Routes Greedily (Sequential) ---
    int id = 0;
    while (true)
    {
        // std::cout << id++ << "\n";

        id++;
        checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));
        find_best_saving_kernel<<<numBlocks, threadsPerBlock>>>(
            d_nodes, d_customer_route_map, d_route_demands, d_route_head, d_route_tail, d_dist_to_depot,
            d_best_saving_out, NUM_CUSTOMERS, CAPACITY);
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
        std::cout << i << " " << j << "\n";

        node_t route_id_i = h_customer_route_map[i];
        node_t route_id_j = h_customer_route_map[j];

        // Check if the merge is valid (different routes and combined demand is within capacity)
        if (route_id_i != route_id_j && h_route_demands[route_id_i] + h_route_demands[route_id_j] <= vrp.capacity)
        {
            node_t head_i = h_route_head[route_id_i];
            node_t tail_i = h_route_tail[route_id_i];
            node_t head_j = h_route_head[route_id_j];
            node_t tail_j = h_route_tail[route_id_j];

            bool merged = false;
            int type = -1;

            if (tail_i == i && head_j == j)
            {

                h_route_tail[route_id_i] = tail_j; // New tail is old tail of j
                merged = true;
                type = 1;
            }
            // Case 2: Tail of route j connects to Head of route i [...j] -> [i...]
            else if (tail_j == j && head_i == i)
            {

                h_route_head[route_id_i] = head_j; // New head is old head of j
                merged = true;
                type = 2;
            }
            // Case 3: Tail of i connects to Tail of j [...i] -> [...j](reversed)
            else if (tail_i == i && tail_j == j)
            {

                h_route_tail[route_id_i] = head_j; // New tail is old head of j
                merged = true;
                type = 3;
            }
            // Case 4: Head of i connects to Head of j [i...](reversed) <- [j...]
            else if (head_i == i && head_j == j)
            {

                h_route_head[route_id_i] = tail_j; // New head is old tail of j
                merged = true;
                type = 4;
            }

            if (merged)
            {

                h_route_demands[route_id_i] += h_route_demands[route_id_j];
                h_route_demands[route_id_j] = 0;
                h_customer_route_map[j] = route_id_i;
                h_route_head[route_id_j] = DEPOT;
                void *args[] = {
                    (void *)&type,
                    &d_customer_route_map,
                    &d_route_demands,
                    &d_route_head,
                    &d_route_tail,
                    &d_next_customer,
                    &d_prev_customer,
                    &d_temporary,
                    (void *)&NUM_CUSTOMERS,
                    (void *)&i,
                    (void *)&j,
                    (void *)&route_id_i,
                    (void *)&route_id_j,
                    (void *)&head_i,
                    (void *)&tail_i,
                    (void *)&head_j,
                    (void *)&tail_j};

                checkCudaErrors(cudaLaunchCooperativeKernel(
                    (void *)update_gpu_mempory,
                    numBlocks1D,
                    threads_per_block,
                    args));
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }
    }
    std::cout << "loop ended\n";

    checkCudaErrors(cudaMemcpy(h_customer_route_map.data(), d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_route_demands.data(), d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_route_head.data(), d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_route_tail.data(), d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_next_customer.data(), d_next_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_prev_customer.data(), d_prev_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));

    // --- 5. Finalize Routes ---
    std::vector<std::vector<node_t>> final_routes;
    std::vector<bool> visited_routes(vrp.size, false);

    for (node_t i = 1; i < vrp.size; ++i)
    {
        node_t route_id = h_customer_route_map[i];
        if (route_id != DEPOT && !visited_routes[route_id])
        {
            visited_routes[route_id] = true;
            std::vector<node_t> current_route;
            node_t current_node = h_route_head[route_id];
            while (current_node != DEPOT)
            {
                current_route.push_back(current_node);
                current_node = h_next_customer[current_node];
            }
            if (!current_route.empty())
            {
                final_routes.push_back(current_route);
            }
        }
    }

    checkCudaErrors(cudaFree(d_nodes));
    checkCudaErrors(cudaFree(d_customer_route_map));
    checkCudaErrors(cudaFree(d_route_demands));
    checkCudaErrors(cudaFree(d_best_saving_out));
    checkCudaErrors(cudaFree(d_dist_to_depot));
    checkCudaErrors(cudaFree(d_route_head));
    checkCudaErrors(cudaFree(d_route_tail));
    checkCudaErrors(cudaFree(d_next_customer));
    checkCudaErrors(cudaFree(d_prev_customer));
    checkCudaErrors(cudaFree(d_temporary));
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
    std::cout << "Number of Routes:   " << postRoutes.size() << std::endl;
    std::cout << "Total Time Taken:    " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID") << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}
