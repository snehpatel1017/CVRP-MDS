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

bool verify_sol(const VRP &vrp, vector<vector<node_t>> final_routes, unsigned capacity)
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

__device__ volatile unsigned int global_counter = 0;
// __device__ volatile unsigned long long int global_counter_2 = 0;

__global__ void find_crush(
    const Point *nodes,
    const node_t *customer_route_map,
    const weight_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    const weight_t *dist_to_depot,
    node_t *crush,
    demand_t capacity,
    unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i <= limit; i += total_threads)
    {
        if (i == 0)
            continue;

        int favourite = -1;
        double best_saving = 0.0;
        demand_t final_cap = DBL_MAX;
        node_t route_id_i = customer_route_map[i];
        demand_t my_demands = route_demands[route_id_i];
        node_t front_i = route_head[route_id_i];
        node_t back_i = route_tail[route_id_i];
        if (front_i == DEPOT || back_i == DEPOT)
            continue;
        for (int j = 1; j <= limit; j++)
        {
            if (j == i)
                continue;

            node_t route_id_j = customer_route_map[j];
            node_t front_j = route_head[route_id_j];
            node_t back_j = route_tail[route_id_j];
            demand_t j_demand = route_demands[route_id_j];
            if (my_demands + j_demand > capacity)
                continue;
            if (front_j == DEPOT || back_j == DEPOT)
                continue;

            double saving_1 = dist_to_depot[back_i] + dist_to_depot[front_j] - device_euclidean_dist(nodes[back_i], nodes[front_j]);
            double saving_2 = dist_to_depot[back_j] + dist_to_depot[front_i] - device_euclidean_dist(nodes[back_j], nodes[front_i]);
            if (best_saving < saving_1)
            {
                best_saving = saving_1;
                favourite = j;
                final_cap = my_demands + j_demand;
            }
            if (best_saving < saving_2)
            {
                best_saving = saving_2;
                favourite = j;
                final_cap = my_demands + j_demand;
            }
            if (best_saving == saving_1 || best_saving == saving_2)
            {
                if (favourite == -1)
                {
                    favourite = j;
                    continue;
                }
                if (final_cap > my_demands + j_demand)
                {
                    favourite = j;
                    final_cap = my_demands + j_demand;
                    continue;
                }
                favourite = min(favourite, j);
            }
        }

        crush[i] = favourite;
        // if (favourite != -1)
        // printf("%d has crush : %d with saving value : %f, and my demands is this : %f, and crush's demands is %f\n", i, crush[i], best_saving, my_demands, route_demands[customer_route_map[favourite]]);
    }
}

__global__ void engagement(
    const Point *nodes,
    node_t *customer_route_map,
    node_t *route_head,
    node_t *route_tail,
    node_t *crush,
    const weight_t *dist_to_depot,
    node_t *store_i,
    node_t *store_j,
    unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i <= last_index; i += total_threads)
    {
        if (crush[i] == -1)
            continue;

        int j = crush[i];
        if (crush[j] != i)
            continue;

        node_t route_id_i = customer_route_map[i];
        node_t route_id_j = customer_route_map[j];
        node_t head_i = route_head[route_id_i];
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        double saving_1 = dist_to_depot[tail_i] + dist_to_depot[head_j] - device_euclidean_dist(nodes[tail_i], nodes[head_j]);
        double saving_2 = dist_to_depot[tail_j] + dist_to_depot[head_i] - device_euclidean_dist(nodes[tail_j], nodes[head_i]);
        if (saving_1 < saving_2)
        {
            continue;
        }
        if (saving_1 == saving_2)
        {
            if (i > j)
                continue;
        }
        int old_pos = atomicAdd((unsigned int *)&global_counter, (unsigned int)1);
        store_i[old_pos] = i;
        store_j[old_pos] = j;
    }
}

__global__ void merging(
    node_t *store_i,
    node_t *store_j,
    node_t *customer_route_map,
    weight_t *route_demands,
    node_t *route_head,
    node_t *route_tail,
    node_t *next_customer,
    node_t *prev_customer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int curr = tid; curr < global_counter; curr += total_threads)
    {
        node_t i = store_i[curr];
        node_t j = store_j[curr];

        node_t route_id_i = customer_route_map[i];
        node_t route_id_j = customer_route_map[j];
        // printf("%d is i and %d is j\n", route_id_i, route_id_j);
        // printf("%f is demand of i and %f is demand of j\n", route_demands[route_id_i], route_demands[route_id_j]);
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        next_customer[tail_i] = head_j;
        prev_customer[head_j] = tail_i;
        route_tail[route_id_i] = tail_j; // New tail is old tail of j
        route_demands[route_id_i] += route_demands[route_id_j];
        route_demands[route_id_j] = 0;
        // printf("%f is new demand of i and %f is new demand of j\n", route_demands[route_id_i], route_demands[route_id_j]);
        route_head[route_id_j] = DEPOT;
        route_tail[route_id_j] = DEPOT;
        customer_route_map[j] = -1;
    }
    // if (tid == 0)
    // {
    //     for (int i = 0; i <= 5; i++)
    //     {
    //         printf("from cuda side %d has demands %f\n", i, route_demands[i]);
    //     }
    // }
}

__global__ void cleanup(
    node_t *customer_route_map,
    node_t *crush,
    unsigned int last_index,
    unsigned int *slow_pointer)
{
    *slow_pointer = 0;
    global_counter = 0;
    for (int i = 1; i <= last_index; i++)
    {
        if (customer_route_map[i] != -1)
        {
            (*slow_pointer)++;
            customer_route_map[*slow_pointer] = customer_route_map[i];
        }
        crush[i] = -1;
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
    std::vector<node_t> h_crush(vrp.size, -1);
    unsigned int h_slow_pointer = NUM_CUSTOMERS;

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
    weight_t *d_dist_to_depot;
    node_t *d_next_customer;
    node_t *d_prev_customer;
    node_t *d_crush;
    node_t *d_store_i;
    node_t *d_store_j;
    unsigned int *d_slow_pointer;

    dim3 threadsPerBlock(1024);
    dim3 numBlocks(56);
    // long long totalThreads = threadsPerBlock.x * numBlocks.x;

    checkCudaErrors(cudaMalloc(&d_nodes, (NUM_CUSTOMERS + 1) * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_next_customer, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_prev_customer, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_crush, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_i, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_j, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_slow_pointer, sizeof(unsigned int)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_nodes, vrp.node.data(), (NUM_CUSTOMERS + 1) * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_head, h_route_head.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_tail, h_route_tail.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_next_customer, h_next_customer.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_prev_customer, h_prev_customer.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_crush, h_crush.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slow_pointer, &h_slow_pointer, sizeof(unsigned int), cudaMemcpyHostToDevice));

    int id = 0;
    unsigned int last_index = NUM_CUSTOMERS;

    while (true)
    {
        // std::cout << id++ << "\n";
        id++;
        // auto st = std::chrono::high_resolution_clock::now();

        find_crush<<<numBlocks, threadsPerBlock>>>(
            d_nodes,
            d_customer_route_map,
            d_route_demands,
            d_route_head,
            d_route_tail,
            d_dist_to_depot,
            d_crush,
            CAPACITY,
            last_index);

        engagement<<<numBlocks, threadsPerBlock>>>(
            d_nodes,
            d_customer_route_map,
            d_route_head,
            d_route_tail,
            d_crush,
            d_dist_to_depot,
            d_store_i,
            d_store_j,
            last_index);

        merging<<<numBlocks, threadsPerBlock>>>(
            d_store_i,
            d_store_j,
            d_customer_route_map,
            d_route_demands,
            d_route_head,
            d_route_tail,
            d_next_customer,
            d_prev_customer);

        cleanup<<<1, 1>>>(
            d_customer_route_map,
            d_crush,
            last_index,
            d_slow_pointer);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(&h_slow_pointer, d_slow_pointer, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        std::cout << h_slow_pointer << " , " << last_index << "\n";
        if (h_slow_pointer == last_index)
        {
            std::cout << "No more positive savings found. Halting." << std::endl;
            std::cout << id << "\n";
            break; // Exit the while loop
        }
        last_index = h_slow_pointer;
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

    for (node_t i = 1; i <= last_index; ++i)
    {
        node_t route_id = h_customer_route_map[i];

        visited_routes[route_id] = true;
        std::vector<node_t> current_route;
        node_t current_node = h_route_head[route_id];
        while (current_node != DEPOT)
        {
            // std::cout << current_node << " , ";
            current_route.push_back(current_node);
            current_node = h_next_customer[current_node];
        }
        if (!current_route.empty())
        {
            final_routes.push_back(current_route);
        }
        // std::cout << "\n";
    }
    std::cout << "routes generated\n";

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