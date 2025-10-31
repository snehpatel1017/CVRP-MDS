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

bool verify_sol(const VRP &vrp, std::vector<std::vector<node_t>> final_routes, unsigned capacity)
{
    /* verifies if the solution is valid or not */
    /**
     * 1. All vertices appear in the solution exactly once.
     * 2. For every route, the capacity constraint is respected.
     **/

    unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
    memset(hist, 0, sizeof(unsigned) * vrp.getSize());

    for (unsigned i = 0; i < final_routes.size(); ++i)
    {
        unsigned route_sum_of_demands = 0;
        for (unsigned j = 0; j < final_routes[i].size(); ++j)
        {
            //~ route_sum_of_demands += points.demands[final_routes[i][j]];
            route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
            hist[final_routes[i][j]] += 1;
        }
        if (route_sum_of_demands > capacity)
        {
            return false;
        }
    }

    for (unsigned i = 1; i < vrp.getSize(); ++i)
    {
        if (hist[i] > 1)
        {
            return false;
        }
        if (hist[i] == 0)
        {
            return false;
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

__device__ int get_pair_mathematical(long long global_index, int n)
{
    double n_double = (double)(n);
    int current_i = (int)(n_double - 2.0 -
                          floor(sqrt(-8.0 * global_index + 4.0 * n_double * (n_double - 1.0) - 7.0) / 2.0 - 0.5));

    return current_i;
}

// __device__ volatile unsigned long long int global_counter_1 = 0;
// __device__ volatile unsigned long long int global_counter_2 = 0;

__global__ void find_best_saving_kernel(
    const Point *nodes,
    const node_t *customer_route_map,
    const demand_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    const weight_t *dist_to_depot,
    node_t *best_saving_i_storage,
    node_t *best_saving_j_storage,
    weight_t *best_saving_value_storage,
    Saving *best_saving_out,
    int num_customers,
    demand_t capacity)
{

    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned long long n = (unsigned long long)num_customers + 1;
    unsigned long long total_pairs = (unsigned long long)num_customers * (num_customers + 1) / 2;

    double local_best_saving_value = 0.0;
    int local_best_i = -1;
    int local_best_j = -1;

    for (long long curr = tid; curr < total_pairs; curr += total_threads)
    {
        int i = get_pair_mathematical(curr, num_customers + 1);

        long long first_pair_index_for_i = (long long)i * n -
                                           (long long)i * (i + 1) / 2;
        int j = curr - first_pair_index_for_i + i + 1;
        // --- 2. Check validity of merging pair (i, j) ---
        node_t route_id_i = customer_route_map[i];
        node_t route_id_j = customer_route_map[j];

        if ((route_id_i == route_id_j) || (route_id_i < 0) || (route_id_j < 0))
            continue;
        if (route_demands[route_id_i] + route_demands[route_id_j] > capacity)
            continue;

        node_t front_i = route_head[route_id_i];
        node_t back_i = route_tail[route_id_i];
        node_t front_j = route_head[route_id_j];
        node_t back_j = route_tail[route_id_j];
        if (front_i == DEPOT || back_i == DEPOT || front_j == DEPOT || back_j == DEPOT)
            continue;

        // --- 3. Calculate saving if the merge is valid ---
        double saving_value_1 = dist_to_depot[back_i]                                   // dist(i, depot)
                                + dist_to_depot[front_j]                                // dist(j, depot)
                                - device_euclidean_dist(nodes[back_i], nodes[front_j]); // dist(i, j)
        double saving_value_2 = dist_to_depot[back_j]                                   // dist(j, depot)
                                + dist_to_depot[front_i]                                // dist(i, depot)
                                - device_euclidean_dist(nodes[back_j], nodes[front_i]); // dist(j, i)
        if (local_best_saving_value < saving_value_1)
        {
            local_best_saving_value = saving_value_1;
            local_best_i = i;
            local_best_j = j;
        }
        if (local_best_saving_value < saving_value_2)
        {
            local_best_saving_value = saving_value_2;
            local_best_i = j;
            local_best_j = i;
        }
    }
    best_saving_i_storage[tid] = local_best_i;
    best_saving_j_storage[tid] = local_best_j;
    best_saving_value_storage[tid] = local_best_saving_value;

    unsigned long long int *global_max_addr_ull = (unsigned long long int *)&(best_saving_out->value);

    unsigned long long int new_val_ull = __double_as_longlong(local_best_saving_value);

    atomicMax(global_max_addr_ull, new_val_ull);
}

__global__ void update_best_node_i(
    node_t *best_saving_i_storage,
    weight_t *best_saving_value_storage,
    Saving *best_saving_out)
{
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_best_i = best_saving_i_storage[tid];
    if (best_saving_value_storage[tid] == best_saving_out->value)
    {
        atomicMin((unsigned int *)&(best_saving_out->i), (unsigned int)local_best_i);
    }
}

__global__ void update_best_node_j(
    node_t *best_saving_i_storage,
    node_t *best_saving_j_storage,
    weight_t *best_saving_value_storage,
    Saving *best_saving_out)
{
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_best_i = best_saving_i_storage[tid];
    int local_best_j = best_saving_j_storage[tid];
    double local_best_saving_value = best_saving_value_storage[tid];
    if ((best_saving_out->i == (unsigned int)local_best_i) && (local_best_saving_value == best_saving_out->value))
    {
        atomicMin((unsigned int *)&(best_saving_out->j), (unsigned int)local_best_j);
        // printf("%lld  \n", tid);
    }
}

__global__ void update_gpu_mempory(
    int last_pointer,
    node_t *customer_route_map,
    demand_t *route_demands,
    node_t *route_head,
    node_t *route_tail,
    node_t *next_customer,
    node_t *prev_customer,
    Saving *best_saving_out,
    int num_customers)
{
    // cg::grid_group grid = cg::this_grid();
    node_t i = best_saving_out->i;
    node_t j = best_saving_out->j;
    node_t route_id_i = customer_route_map[i];
    node_t route_id_j = customer_route_map[j];
    node_t tail_i = route_tail[route_id_i];
    node_t head_j = route_head[route_id_j];
    node_t tail_j = route_tail[route_id_j];
    next_customer[tail_i] = head_j;
    prev_customer[head_j] = tail_i;
    route_tail[route_id_i] = tail_j; // New tail is old tail of j
    route_demands[route_id_i] += route_demands[route_id_j];
    route_demands[route_id_j] = 0;
    route_head[route_id_j] = DEPOT;
    route_tail[route_id_j] = DEPOT;
    customer_route_map[j] = customer_route_map[last_pointer];
    best_saving_out->value = 0;
    best_saving_out->i = INT_MAX;
    best_saving_out->j = INT_MAX;
}

std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp)
{
    const int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    const demand_t CAPACITY = vrp.getCapacity();
    std::cout << "Number of Customers: " << NUM_CUSTOMERS << "\n";

    std::vector<node_t> h_customer_route_map(NUM_CUSTOMERS + 1);
    std::vector<demand_t> h_route_demands(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_head(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_tail(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_next_customer(vrp.size, DEPOT);
    std::vector<node_t> h_prev_customer(vrp.size, DEPOT);
    std::vector<node_t> temporary(vrp.size, DEPOT);

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
    node_t *d_best_saving_i_storage;
    node_t *d_best_saving_j_storage;
    weight_t *d_best_saving_value_storage;
    Saving *d_best_saving_out;
    weight_t *d_dist_to_depot;
    node_t *d_next_customer;
    node_t *d_prev_customer;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((int)prop.multiProcessorCount);
    long long totalThreads = threadsPerBlock.x * numBlocks.x;

    checkCudaErrors(cudaMalloc(&d_nodes, (NUM_CUSTOMERS + 1) * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_out, sizeof(Saving)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_i_storage, (totalThreads) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_j_storage, (totalThreads) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving_value_storage, (totalThreads) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_next_customer, vrp.size * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_prev_customer, vrp.size * sizeof(node_t)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_nodes, vrp.node.data(), (NUM_CUSTOMERS + 1) * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_head, h_route_head.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_tail, h_route_tail.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_next_customer, h_next_customer.data(), vrp.size * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_prev_customer, h_prev_customer.data(), vrp.size * sizeof(node_t), cudaMemcpyHostToDevice));
    // Initialize the output struct on the GPU to a known "worst" state
    Saving h_best_saving_init = {INT_MAX, INT_MAX, 0};
    checkCudaErrors(cudaMemcpy(d_best_saving_out, &h_best_saving_init, sizeof(Saving), cudaMemcpyHostToDevice));

    // --- 4. KERNEL LAUNCH ---

    Saving h_result;
    int id = 0;
    int last_pointer = NUM_CUSTOMERS;
    std::chrono::time_point<std::chrono::high_resolution_clock> st, en;
    while (true)
    {

        id++;
        if (id == 1)
            st = std::chrono::high_resolution_clock::now();
        find_best_saving_kernel<<<numBlocks, threadsPerBlock>>>(
            d_nodes, d_customer_route_map, d_route_demands, d_route_head, d_route_tail, d_dist_to_depot,
            d_best_saving_i_storage, d_best_saving_j_storage, d_best_saving_value_storage,
            d_best_saving_out, last_pointer, CAPACITY);

        update_best_node_i<<<numBlocks, threadsPerBlock>>>(
            d_best_saving_i_storage, d_best_saving_value_storage, d_best_saving_out);

        update_best_node_j<<<numBlocks, threadsPerBlock>>>(
            d_best_saving_i_storage, d_best_saving_j_storage, d_best_saving_value_storage, d_best_saving_out);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(&h_result, d_best_saving_out, sizeof(Saving), cudaMemcpyDeviceToHost));

        if (h_result.value <= 1e-6)
        {
            std::cout << "No more positive savings found. Halting." << std::endl;
            std::cout << id << "\n";
            break; // Exit the while loop
        }

        update_gpu_mempory<<<1, 1>>>(
            last_pointer,
            d_customer_route_map,
            d_route_demands,
            d_route_head,
            d_route_tail,
            d_next_customer,
            d_prev_customer,
            d_best_saving_out,
            NUM_CUSTOMERS);
        last_pointer--;
        if (id == 1)
        {
            en = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = en - st;
            std::cout << "Time for first iteration: " << elapsed.count() << " seconds\n";
        }
    }
    std::cout << "loop ended\n";

    checkCudaErrors(cudaMemcpy(h_customer_route_map.data(), d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(h_route_demands.data(), d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_route_head.data(), d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(h_route_tail.data(), d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_next_customer.data(), d_next_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(h_prev_customer.data(), d_prev_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));
    std::cout << "memory copied back to host\n";
    // --- 5. Finalize Routes ---
    std::vector<std::vector<node_t>> final_routes;
    std::vector<bool> visited_routes(vrp.size, false);

    for (node_t i = 1; i <= last_pointer; ++i)
    {
        node_t route_id = h_customer_route_map[i];

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
    std::cout << "routes generated\n";

    // checkCudaErrors(cudaFree(d_nodes));
    // checkCudaErrors(cudaFree(d_customer_route_map));
    // checkCudaErrors(cudaFree(d_route_demands));
    // checkCudaErrors(cudaFree(d_best_saving_out));
    // checkCudaErrors(cudaFree(d_dist_to_depot));
    // checkCudaErrors(cudaFree(d_route_head));
    // checkCudaErrors(cudaFree(d_route_tail));
    // checkCudaErrors(cudaFree(d_next_customer));
    // checkCudaErrors(cudaFree(d_prev_customer));
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
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    weight_t total_cost = calCost(vrp, routes);
    // routes = postProcessIt(vrp, routes, total_cost);
    // total_cost = calCost(vrp, routes);
    bool is_valid = verify_sol(vrp, routes, vrp.getCapacity());
    // for (auto x : routes)
    // {
    //     for (auto y : x)
    //     {
    //         cout << y << ",";
    //     }
    //     cout << "\n";
    // }

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