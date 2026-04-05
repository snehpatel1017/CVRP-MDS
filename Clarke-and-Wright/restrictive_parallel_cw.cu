#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cfloat>  // For DBL_MAX
#include <iomanip> // For std::setprecision
#include <chrono>  // For timing
#include <omp.h>
#include <stdio.h>
#include <utility>
#include <set>
#include "KDTree.hpp"
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
    static bool isRound;
    static bool verbose;
    static int K;

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

bool VRP::isRound = false;
bool VRP::verbose = false;
int VRP::K = 20;

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
    double dist = sqrt(dx * dx + dy * dy);
    if (isRound)
        return std::round(dist);
    return dist;
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

__device__ double device_euclidean_dist(const double aX, const double aY, const double bX, const double bY)
{
    return sqrt((aX - bX) * (aX - bX) + (aY - bY) * (aY - bY));
}

__device__ volatile unsigned int global_counter = 0;
// __device__ volatile unsigned long long int global_counter_2 = 0;
__device__ double atomicMax(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        // Compare double values and use CAS to update if the new value is larger
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));

        // Note: uses integer comparison to detect if the value was changed by another thread
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicMin(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        // Compare double values and use CAS to update if the new value is smaller
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));

        // Uses integer comparison to detect if the memory was modified by another thread
    } while (assumed != old);

    return __longlong_as_double(old);
}
__global__ void k1(
    const node_t *edges_X,
    const node_t *edges_Y,
    const weight_t *edges_W,
    weight_t *best_saving,
    const weight_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    demand_t capacity,
    unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i < limit; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];
        if (edges_W[i] <= 0)
            continue;

        node_t cr = route_head[b];
        if (route_demands[cr] + route_demands[route_head[a]] > capacity)
            continue;

        atomicMax(&best_saving[a], edges_W[i]);
        atomicMax(&best_saving[cr], edges_W[i]);
    }
}

__global__ void k2(
    const node_t *edges_X,
    const node_t *edges_Y,
    const weight_t *edges_W,
    weight_t *best_saving,
    demand_t *best_demand,
    const weight_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    demand_t capacity,
    unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i < limit; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];

        if (edges_W[i] <= 0)
            continue;
        node_t cr = route_head[b];
        demand_t tot = route_demands[cr] + route_demands[route_head[a]];
        if (tot > capacity)
            continue;
        if (best_saving[a] == edges_W[i])
            atomicMin(&best_demand[a], tot);
        if (best_saving[cr] == edges_W[i])
            atomicMin(&best_demand[cr], tot);
    }
}

__global__ void k3(const node_t *edges_X,
                   const node_t *edges_Y,
                   const weight_t *edges_W,
                   weight_t *best_saving,
                   demand_t *best_demand,
                   node_t *crush,
                   const weight_t *route_demands,
                   const node_t *route_head,
                   const node_t *route_tail,
                   demand_t capacity,
                   unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i < limit; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];

        if (edges_W[i] <= 0)
            continue;
        node_t cr = route_head[b];
        demand_t tot = route_demands[cr] + route_demands[route_head[a]];
        if (tot > capacity || best_saving[a] != best_saving[cr] || best_saving[a] != edges_W[i] || best_demand[a] != tot || best_demand[cr] != tot)
            continue;

        atomicMin(&crush[a], cr);
        atomicMin(&crush[cr], a);
    }
}

__global__ void get_pairs(
    const double *X,
    const double *Y,
    node_t *route_head,
    node_t *route_tail,
    node_t *crush,
    const weight_t *dist_to_depot,
    node_t *store_i,
    node_t *store_j,
    unsigned int last_index,
    unsigned int *holding_global_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i <= last_index; i += total_threads)
    {
        if (crush[i] == last_index + 2)
            continue;

        int j = crush[i];
        if (crush[j] != i)
            continue;

        node_t route_id_i = i;
        node_t route_id_j = j;
        node_t head_i = route_head[route_id_i];
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        double saving_1 = dist_to_depot[tail_i] + dist_to_depot[head_j] - device_euclidean_dist(X[tail_i], Y[tail_i], X[head_j], Y[head_j]);
        double saving_2 = dist_to_depot[tail_j] + dist_to_depot[head_i] - device_euclidean_dist(X[tail_j], Y[tail_j], X[head_i], Y[head_i]);
        if (saving_1 < saving_2)
        {
            continue;
        }
        if (saving_1 == saving_2)
        {
            if (i > j)
                continue;
        }
        int old_pos = atomicAdd((unsigned int *)holding_global_counter, (unsigned int)1);
        store_i[old_pos] = i;
        store_j[old_pos] = j;
    }
}

__global__ void merging(
    node_t *store_i,
    node_t *store_j,
    weight_t *route_demands,
    node_t *route_head,
    node_t *route_tail,
    node_t *next_customer, unsigned int *holding_global_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int curr = tid; curr < *holding_global_counter; curr += total_threads)
    {
        node_t i = store_i[curr];
        node_t j = store_j[curr];
        node_t route_id_i = i;
        node_t route_id_j = j;
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        next_customer[tail_i] = head_j;
        route_tail[route_id_i] = tail_j;
        route_head[tail_j] = route_id_i;
        route_demands[route_id_i] += route_demands[route_id_j];
        route_demands[route_id_j] = 0;
        route_head[route_id_j] = route_id_i;
    }
}

__global__ void new_cleanup(
    node_t *edges_X,
    node_t *edges_Y,
    weight_t *edges_W,
    node_t *temp_edges_X,
    node_t *temp_edges_Y,
    weight_t *temp_edges_W,
    const node_t *route_head,
    const node_t *route_tail,
    unsigned int last_index,
    unsigned int *slow_pointer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int i = tid; i < last_index; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];
        if (route_head[a] == route_head[b])
            continue;
        if (a != route_head[a] && route_tail[route_head[a]] != a)
            continue;
        if (b != route_head[b] && route_tail[route_head[b]] != b)
            continue;
        if ((route_head[b] == b && route_tail[route_head[a]] == a) || (route_head[a] == a && route_tail[route_head[b]] == b))
        {
            if ((route_head[b] == b && route_tail[route_head[a]] == a))
            {
                node_t temp = a;

                a = b;
                b = temp;
            }

            unsigned int pos = atomicAdd((unsigned int *)slow_pointer, 1);
            temp_edges_X[pos] = a;
            temp_edges_Y[pos] = b;
            temp_edges_W[pos] = edges_W[i];
        }
    }
}

__global__ void final_cleanup(
    int *lock,
    node_t *crush,
    weight_t *best_saving,
    demand_t *best_demand,
    unsigned int last_index,
    const demand_t CAPACITY,
    unsigned int *holding_global_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (tid == 0)
    {
        *holding_global_counter = 0;

        global_counter = 0;
    }
    for (int i = tid; i <= last_index; i += total_threads)
    {
        if (i == 0)
            continue;
        crush[i] = last_index + 2;
        best_saving[i] = 0;
        best_demand[i] = CAPACITY + 1;
    }
}

std::vector<std::vector<std::pair<node_t, node_t>>> mergings;
std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp, std::vector<std::vector<std::pair<weight_t, node_t>>> &neighbours)
{
    const int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    const demand_t CAPACITY = vrp.getCapacity();
    std::cout << "Number of Customers: " << NUM_CUSTOMERS << "\n";
    std::vector<double> h_X(NUM_CUSTOMERS + 1);
    std::vector<double> h_Y(NUM_CUSTOMERS + 1);
    std::vector<weight_t> best_saving(NUM_CUSTOMERS + 1, 0);
    std::vector<demand_t> best_demand(NUM_CUSTOMERS + 1, CAPACITY + 1);
    std::vector<int> lock(NUM_CUSTOMERS + 1);
    std::vector<demand_t> h_route_demands(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_head(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_tail(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_next_customer(vrp.size, DEPOT);
    std::vector<node_t> edges_X(NUM_CUSTOMERS * vrp.K), edges_Y(vrp.size * vrp.K);
    std::vector<weight_t> edges_W(NUM_CUSTOMERS * vrp.K);
    std::vector<node_t> h_crush(vrp.size, NUM_CUSTOMERS + 2);
    std::vector<node_t> h_store_i((NUM_CUSTOMERS) / 2 + 1, -1);
    std::vector<node_t> h_store_j((NUM_CUSTOMERS) / 2 + 1, -1);
    unsigned int h_slow_pointer = 0;
    unsigned int h_holding_global_counter = 0;
    weight_t h_total_cost = 0.0;

    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
    {
        // Initially, each customer is in their own route
        h_X[i] = vrp.node[i].x;
        h_Y[i] = vrp.node[i].y;
        lock[i] = 0;
        h_route_demands[i] = vrp.node[i].demand;
        h_route_head[i] = i;
        h_route_tail[i] = i;
    }
    int edge_index = 0;
    for (int i = 1; i <= NUM_CUSTOMERS; i++)
    {
        for (int j = 0; j < neighbours[i].size(); j++)
        {
            edges_X[edge_index] = i;
            edges_Y[edge_index] = neighbours[i][j].second * -1;
            edges_W[edge_index] = neighbours[i][j].first;
            edge_index++;
        }
    }

    // --- 2. DEVICE: Allocate GPU memory ---
    double *d_X;
    double *d_Y;
    node_t *d_edges_X, *d_edges_Y;
    node_t *d_temp_edges_X, *d_temp_edges_Y;
    weight_t *d_edges_W, *d_temp_edges_W;
    weight_t *d_best_saving;
    demand_t *d_best_demand;
    int *d_lock;
    demand_t *d_route_demands;
    node_t *d_route_head;
    node_t *d_route_tail;
    weight_t *d_dist_to_depot;
    node_t *d_next_customer;
    weight_t *d_total_cost;
    node_t *d_crush;
    node_t *d_store_i;
    node_t *d_store_j;
    unsigned int *d_slow_pointer;
    unsigned int *d_holding_global_counter;

    dim3 threadsPerBlock(1024);
    dim3 numBlocks((int)(NUM_CUSTOMERS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    // long long totalThreads = threadsPerBlock.x * numBlocks.x; int tid = blockIdx.x * blockDim.x + threadIdx.x;

    checkCudaErrors(cudaMalloc(&d_X, (NUM_CUSTOMERS + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Y, (NUM_CUSTOMERS + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_edges_X, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_temp_edges_X, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_edges_Y, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_temp_edges_Y, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_edges_W, NUM_CUSTOMERS * vrp.K * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_temp_edges_W, NUM_CUSTOMERS * vrp.K * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_best_demand, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_lock, (NUM_CUSTOMERS + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_next_customer, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_total_cost, sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_crush, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_i, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_j, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_slow_pointer, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_holding_global_counter, sizeof(unsigned int)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_X, h_X.data(), (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Y, h_Y.data(), (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_edges_X, edges_X.data(), edges_X.size() * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_edges_Y, edges_Y.data(), edges_X.size() * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_edges_W, edges_W.data(), edges_X.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_best_saving, best_saving.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_best_demand, best_demand.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lock, lock.data(), (NUM_CUSTOMERS + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_crush, h_crush.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_head, h_route_head.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_tail, h_route_tail.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_next_customer, h_next_customer.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_crush, h_crush.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slow_pointer, &h_slow_pointer, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_total_cost, &h_total_cost, sizeof(weight_t), cudaMemcpyHostToDevice));

    int id = 0;
    unsigned int last_index = edges_X.size();
    std::chrono::time_point<std::chrono::high_resolution_clock> st, en;

    while (true)
    {

        id++;
        if (id == 1)
            st = std::chrono::high_resolution_clock::now();

        
        k1<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_best_saving,
            d_route_demands,
            d_route_head,
            d_route_tail,
            CAPACITY,
            last_index);
        /*checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(best_saving.data(), d_best_saving, (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyDeviceToHost));
        for (int i = 1; i <= NUM_CUSTOMERS; i++)
        {
            std::cout << best_saving[i] << " | ";
        }
        std::cout << "\n";*/
        k2<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_best_saving,
            d_best_demand,
            d_route_demands,
            d_route_head,
            d_route_tail,
            CAPACITY,
            last_index);
        /*checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(best_demand.data(), d_best_demand, (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyDeviceToHost));
        for (int i = 1; i <= NUM_CUSTOMERS; i++)
        {
            std::cout << best_demand[i] << " | ";
        }
        std::cout << "\n";*/
        k3<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_best_saving,
            d_best_demand,
            d_crush,
            d_route_demands,
            d_route_head,
            d_route_tail,
            CAPACITY,
            last_index);
        /*
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(h_crush.data(), d_crush, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
            std::cout << "Crush array: \n";
            for (int i = 1; i <= NUM_CUSTOMERS; i++)
            {
                std::cout << h_crush[i] << " | ";
            }
            std::cout << "\n";*/
        get_pairs<<<numBlocks, threadsPerBlock>>>(
            d_X,
            d_Y,
            d_route_head,
            d_route_tail,
            d_crush,
            d_dist_to_depot,
            d_store_i,
            d_store_j,
            NUM_CUSTOMERS,
            d_holding_global_counter);
        if (vrp.verbose)
        {

            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(h_route_tail.data(), d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&h_holding_global_counter, d_holding_global_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_store_i.data(), d_store_i, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_store_j.data(), d_store_j, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
            std::cout << "Merging pairs: \n";
            for (int i = 0; i < h_holding_global_counter; i++)
            {
                std::cout << h_route_tail[h_store_i[i]] << " , " << h_store_j[i] << "\n";
            }
        }

        merging<<<numBlocks, threadsPerBlock>>>(
            d_store_i,
            d_store_j,
            d_route_demands,
            d_route_head,
            d_route_tail,
            d_next_customer, d_holding_global_counter);

        // cleanup<<<1, 1>>>(
        //     d_customer_route_map,
        //     d_crush,
        //     last_index,
        //     d_slow_pointer);

        new_cleanup<<<numBlocks, threadsPerBlock>>>(d_edges_X,
                                                    d_edges_Y,
                                                    d_edges_W,
                                                    d_temp_edges_X,
                                                    d_temp_edges_Y,
                                                    d_temp_edges_W,
                                                    d_route_head,
                                                    d_route_tail,
                                                    last_index,
                                                    d_slow_pointer);

        final_cleanup<<<numBlocks, threadsPerBlock>>>(
            d_lock,
            d_crush,
            d_best_saving,
            d_best_demand,
            NUM_CUSTOMERS,
            CAPACITY,
            d_holding_global_counter);

        checkCudaErrors(cudaDeviceSynchronize());
        // checkCudaErrors(cudaMemcpy(&h_holding_global_counter, d_holding_global_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&h_slow_pointer, d_slow_pointer, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::swap(d_edges_X, d_temp_edges_X);
        std::swap(d_edges_Y, d_temp_edges_Y);
        std::swap(d_edges_W, d_temp_edges_W);
        // std::cout << h_slow_pointer << " , " << last_index << "\n";
        if (last_index == h_slow_pointer)
        {
            std::cout << "No more positive savings found. Halting." << std::endl;
            std::cout << id << "\n";
            break; // Exit the while loop
        }
        last_index = h_slow_pointer;
        h_slow_pointer = 0;
        checkCudaErrors(cudaMemcpy(d_slow_pointer, &h_slow_pointer, sizeof(unsigned int), cudaMemcpyHostToDevice));

        if (id == 1)
        {
            en = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = en - st;
            std::cout << "Time for first iteration: " << elapsed.count() << " seconds\n";
        }
    }
    std::cout << "loop ended\n";

    // checkCudaErrors(cudaMemcpy(h_route_demands.data(), d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_route_head.data(), d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(h_route_tail.data(), d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_next_customer.data(), d_next_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(h_prev_customer.data(), d_prev_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));
    std::cout << "memory copied back to host\n";
    // --- 5. Finalize Routes ---
    std::vector<std::vector<node_t>> final_routes;

    for (node_t i = 1; i <= NUM_CUSTOMERS; ++i)
    {

        node_t current_node = h_route_head[i];
        if (current_node != i)
            continue;
        std::vector<node_t> current_route;
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

void tsp_approx(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, node_t ncities)
{
    node_t i, j;
    node_t ClosePt = 0;
    weight_t CloseDist;

    //~ node_t endtour=0;

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
        /*swapping tour[i] and tour[ClosePt]*/
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
        // postprocessing solRoutes[i]
        unsigned sz = solRoutes[i].size();
        std::vector<node_t> cities(sz + 1);
        std::vector<node_t> tour(sz + 1);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = solRoutes[i][j];

        cities[sz] = 0; // the last node is the depot.

        tsp_approx(vrp, cities, tour, sz + 1);

        // the first element of the tour is now the depot. So, ignore tour[0] and insert the rest into the vector.

        std::vector<node_t> curr_route;
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
    // 'cities' contains the original solution. It is updated during the course of the 2opt-scheme to contain the 2opt soln.
    // 'tour' is an auxillary array.

    // repeat until no improvement is made
    unsigned improve = 0;

    while (improve < 2)
    {
        double best_distance = 0.0;
        //~ best_distance += L2_dist(points.x_coords[cities[0]], points.y_coords[cities[0]], 0, 0); // computing distance of the first point in the route with the depot.
        best_distance += vrp.get_dist(DEPOT, cities[0]); // computing distance of the first point in the route with the depot.

        for (unsigned jj = 1; jj < ncities; ++jj)
        {
            //~ best_distance += L2_dist(points.x_coords[cities[jj-1]], points.y_coords[cities[jj-1]], points.x_coords[cities[jj]], points.y_coords[cities[jj]]);
            best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
        }
        //~ best_distance += L2_dist(points.x_coords[cities[ncities-1]], points.y_coords[cities[ncities-1]], 0, 0); // computing distance of the last point in the route with the depot.
        best_distance += vrp.get_dist(DEPOT, cities[ncities - 1]);
        // 1x 2x 3x 4 5
        //  1 2  3  4 5
        for (unsigned i = 0; i < ncities - 1; i++)
        {
            for (unsigned k = i + 1; k < ncities; k++)
            {
                for (unsigned c = 0; c < i; ++c)
                {
                    tour[c] = cities[c];
                }

                unsigned dec = 0;
                for (unsigned c = i; c < k + 1; ++c)
                {
                    tour[c] = cities[k - dec];
                    dec++;
                }

                for (unsigned c = k + 1; c < ncities; ++c)
                {
                    tour[c] = cities[c];
                }
                double new_distance = 0.0;
                //~ new_distance += L2_dist(points.x_coords[tour[0]], points.y_coords[tour[0]], 0, 0); // computing distance of the first point in the route with the depot.
                new_distance += vrp.get_dist(DEPOT, tour[0]);
                for (unsigned jj = 1; jj < ncities; ++jj)
                {
                    //~ new_distance += L2_dist(points.x_coords[tour[jj-1]], points.y_coords[tour[jj-1]], points.x_coords[tour[jj]], points.y_coords[tour[jj]]);
                    new_distance += vrp.get_dist(tour[jj - 1], tour[jj]);
                }
                //~ new_distance += L2_dist(points.x_coords[tour[ncities-1]], points.y_coords[tour[ncities-1]], 0, 0); // computing distance of the last point in the route with the depot.
                new_distance += vrp.get_dist(DEPOT, tour[ncities - 1]);

                if (new_distance < best_distance)
                {
                    // Improvement found so reset
                    improve = 0;
                    for (unsigned jj = 0; jj < ncities; jj++)
                        cities[jj] = tour[jj];
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
        // postprocessing final_routes[i]
        unsigned sz = final_routes[i].size();
        //~ unsigned* cities = (unsigned*) malloc(sizeof(unsigned) * (sz));
        //~ unsigned* tour = (unsigned*) malloc(sizeof(unsigned) * (sz));  // this is an auxillary array

        std::vector<node_t> cities(sz);
        std::vector<node_t> tour(sz);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = final_routes[i][j];

        std::vector<node_t> curr_route;

        if (sz > 2)                          // for sz <= 1, the cost of the path cannot change. So no point running this.
            tsp_2opt(vrp, cities, tour, sz); // MAIN

        for (unsigned kk = 0; kk < sz; ++kk)
        {
            curr_route.push_back(cities[kk]);
        }

        postprocessed_final_routes.push_back(curr_route);
    }
    return postprocessed_final_routes;
}

weight_t get_total_cost_of_routes(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes)
{
    weight_t total_cost = 0.0;
    for (unsigned ii = 0; ii < final_routes.size(); ++ii)
    {
        weight_t curr_route_cost = 0;
        //~ curr_route_cost += L2_dist(points.x_coords[final_routes[ii][0]], points.y_coords[final_routes[ii][0]], 0, 0); // computing distance of the first point in the route with the depot.
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
        for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
        {
            //~ curr_route_cost += L2_dist(points.x_coords[final_routes[ii][jj-1]], points.y_coords[final_routes[ii][jj-1]], points.x_coords[final_routes[ii][jj]], points.y_coords[final_routes[ii][jj]]);
            curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
        }
        //~ curr_route_cost += L2_dist(points.x_coords[final_routes[ii][final_routes[ii].size()-1]], points.y_coords[final_routes[ii][final_routes[ii].size()-1]], 0, 0); // computing distance of the last point in the route with the depot.
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);

        total_cost += curr_route_cost;
    }

    return total_cost;
}

//
// MAIN POST PROCESS ROUTINE
//
std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t &minCost)
{
    std::vector<std::vector<node_t>> postprocessed_final_routes;

    auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
    auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
    auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

//~ weight_t postprocessed_final_routes_cost;
#pragma omp parallel for
    for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz)
    {
        // include the better route between postprocessed_final_routes2[zzz] and postprocessed_final_routes3[zzz] in the final solution.

        std::vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
        std::vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];

        unsigned sz2 = postprocessed_route2.size();
        unsigned sz3 = postprocessed_route3.size();

        // finding the cost of postprocessed_route2

        weight_t postprocessed_route2_cost = 0.0;
        //~ postprocessed_route2_cost += L2_dist(points.x_coords[postprocessed_route2[0]], points.y_coords[postprocessed_route2[0]], 0, 0); // computing distance of the first point in the route with the depot.
        postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]); // computing distance of the first point in the route with the depot.
        for (unsigned jj = 1; jj < sz2; ++jj)
        {
            //~ postprocessed_route2_cost += L2_dist(points.x_coords[postprocessed_route2[jj-1]], points.y_coords[postprocessed_route2[jj-1]], points.x_coords[postprocessed_route2[jj]], points.y_coords[postprocessed_route2[jj]]);
            postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
        }
        //~ postprocessed_route2_cost += L2_dist(points.x_coords[postprocessed_route2[sz2-1]], points.y_coords[postprocessed_route2[sz2-1]], 0, 0); // computing distance of the last point in the route with the depot.
        postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

        // finding the cost of postprocessed_route3

        weight_t postprocessed_route3_cost = 0.0;
        //~ postprocessed_route3_cost += L2_dist(points.x_coords[postprocessed_route3[0]], points.y_coords[postprocessed_route3[0]], 0, 0); // computing distance of the first point in the route with the depot.
        postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
        for (unsigned jj = 1; jj < sz3; ++jj)
        {
            //~ postprocessed_route3_cost += L2_dist(points.x_coords[postprocessed_route3[jj-1]], points.y_coords[postprocessed_route3[jj-1]], points.x_coords[postprocessed_route3[jj]], points.y_coords[postprocessed_route3[jj]]);
            postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
        }
        //~ postprocessed_route3_cost += L2_dist(points.x_coords[postprocessed_route3[sz3-1]], points.y_coords[postprocessed_route3[sz3-1]], 0, 0); // computing distance of the last point in the route with the depot.
        postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

        // postprocessed_route2_cost is lower
        if (postprocessed_route3_cost > postprocessed_route2_cost)
        {
            postprocessed_final_routes.push_back(postprocessed_route2);
        }
        // postprocessed_route3_cost is lower
        else
        {
            postprocessed_final_routes.push_back(postprocessed_route3);
        }
    }

    auto postprocessed_final_routes_cost = get_total_cost_of_routes(vrp, postprocessed_final_routes);

    minCost = postprocessed_final_routes_cost;

    return postprocessed_final_routes;
}
std::string get_base_name(const std::string &path)
{
    // Find the last path separator (works for both / and \)
    size_t last_slash_idx = path.find_last_of("/\\");
    std::string filename = (last_slash_idx == std::string::npos) ? path : path.substr(last_slash_idx + 1);

    // Find the last period (to remove extension)
    size_t period_idx = filename.rfind('.');
    if (period_idx != std::string::npos && period_idx != 0)
    { // Avoid cases like ".hiddenfile"
        filename = filename.substr(0, period_idx);
    }

    return filename;
}

// O(1) Capacity Check using precomputed prefix loads
inline bool isValidCapacity(double loadR1_firstHalf, double loadR2_secondHalf,
                            double loadR2_firstHalf, double loadR1_secondHalf, double maxCapacity)
{
    return (loadR1_firstHalf + loadR2_secondHalf <= maxCapacity) &&
           (loadR2_firstHalf + loadR1_secondHalf <= maxCapacity);
}

// Optimized 2-Opt* with Neighborhood Restriction and On-The-Fly Distances
// Uses a template so you can pass your custom 'vrp' object directly

void runGranular2OptStar(std::vector<std::vector<int>> &routes,
                         VRP &vrp,
                         const std::vector<std::vector<int>> &nearestNeighbors, // The N x K matrix
                         double maxCapacity,
                         int numCustomers)
{
    std::cout << "Starting Granular 2-Opt*...\n";
    bool improvement = true;

    while (improvement)
    {
        improvement = false;

        // 1. PRECOMPUTE LOOKUPS: O(n) time
        // nodeToRoute maps customer ID -> which route they are in
        // nodeToIndex maps customer ID -> their index within that route
        std::vector<int> nodeToRoute(numCustomers + 1, -1);
        std::vector<int> nodeToIndex(numCustomers + 1, -1);
        std::vector<std::vector<double>> prefixLoads(routes.size());

        for (size_t r = 0; r < routes.size(); ++r)
        {
            double currentLoad = 0.0;
            for (size_t i = 0; i < routes[r].size(); ++i)
            {
                int node = routes[r][i];
                nodeToRoute[node] = r;
                nodeToIndex[node] = i;

                currentLoad += vrp.node[node].demand;
                prefixLoads[r].push_back(currentLoad);
            }
        }

        // 2. SEARCH MOVES: O(n * K) time
        for (int u = 1; u <= numCustomers; ++u)
        {
            int r1 = nodeToRoute[u];
            if (r1 == -1)
                continue; // Node is not in a route (e.g., depot or unassigned)
            int i = nodeToIndex[u];

            // Only check the closest K neighbors for node U
            for (int v : nearestNeighbors[u])
            {
                int r2 = nodeToRoute[v];

                // Skip if 'v' is unassigned, or if they are in the same route
                if (r2 == -1 || r1 == r2)
                    continue;
                int j = nodeToIndex[v];

                // Node X is after U, Node Y is after V (Use DEPOT if they are the last nodes)
                int x = (i == routes[r1].size() - 1) ? DEPOT : routes[r1][i + 1];
                int y = (j == routes[r2].size() - 1) ? DEPOT : routes[r2][j + 1];

                // Calculate distance difference using your on-the-fly get_dist function
                double currentEdgesDist = vrp.get_dist(u, x) + vrp.get_dist(v, y);
                double newEdgesDist = vrp.get_dist(u, y) + vrp.get_dist(v, x);

                // If no distance saved, skip immediately
                if (newEdgesDist >= currentEdgesDist - 1e-6)
                    continue;

                // O(1) Capacity Check using prefix sums
                double loadR1_firstHalf = prefixLoads[r1][i];
                double loadR1_total = prefixLoads[r1].back();
                double loadR1_secondHalf = loadR1_total - loadR1_firstHalf;

                double loadR2_firstHalf = prefixLoads[r2][j];
                double loadR2_total = prefixLoads[r2].back();
                double loadR2_secondHalf = loadR2_total - loadR2_firstHalf;

                if (!isValidCapacity(loadR1_firstHalf, loadR2_secondHalf, loadR2_firstHalf, loadR1_secondHalf, maxCapacity))
                {
                    continue; // Move violates capacity
                }

                // --- APPLY THE MOVE ---
                std::vector<int> newRoute1;
                std::vector<int> newRoute2;
                newRoute1.reserve(i + 1 + (routes[r2].size() - j - 1));
                newRoute2.reserve(j + 1 + (routes[r1].size() - i - 1));

                // Rebuild Route 1
                for (int k = 0; k <= i; ++k)
                    newRoute1.push_back(routes[r1][k]);
                for (int k = j + 1; k < routes[r2].size(); ++k)
                    newRoute1.push_back(routes[r2][k]);

                // Rebuild Route 2
                for (int k = 0; k <= j; ++k)
                    newRoute2.push_back(routes[r2][k]);
                for (int k = i + 1; k < routes[r1].size(); ++k)
                    newRoute2.push_back(routes[r1][k]);

                routes[r1] = newRoute1;
                routes[r2] = newRoute2;

                improvement = true;
                goto restart_search; // Restart to refresh prefix sums and lookups
            }
        }
    restart_search:;
    }
}

void runGranularRelocate(std::vector<std::vector<int>> &routes,
                         VRP &vrp,
                         const std::vector<std::vector<int>> &nearestNeighbors,
                         double maxCapacity,
                         int numCustomers)
{
    std::cout << "Starting Granular Relocate...\n";
    bool improvement = true;

    while (improvement)
    {
        improvement = false;

        // 1. PRECOMPUTE LOOKUPS: O(n) time
        std::vector<int> nodeToRoute(numCustomers + 1, -1);
        std::vector<int> nodeToIndex(numCustomers + 1, -1);
        std::vector<double> routeLoads(routes.size(), 0.0);

        for (size_t r = 0; r < routes.size(); ++r)
        {
            double currentLoad = 0.0;
            for (size_t i = 0; i < routes[r].size(); ++i)
            {
                int node = routes[r][i];
                nodeToRoute[node] = r;
                nodeToIndex[node] = i;
                currentLoad += vrp.node[node].demand;
            }
            routeLoads[r] = currentLoad;
        }

        // 2. SEARCH MOVES: O(n * K) time
        for (int u = 1; u <= numCustomers; ++u)
        {
            int r1 = nodeToRoute[u];
            if (r1 == -1)
                continue; // Node not assigned
            int i = nodeToIndex[u];

            double demandU = vrp.node[u].demand;

            for (int v : nearestNeighbors[u])
            {
                int r2 = nodeToRoute[v];

                // Skip unassigned nodes, or if U and V are already in the same route
                if (r2 == -1 || r1 == r2)
                    continue;

                // O(1) Capacity Check: Can Route 2 accept U?
                if (routeLoads[r2] + demandU > maxCapacity)
                    continue;

                int j = nodeToIndex[v];

                // Identify neighbors in Route 1 (for removal of U)
                int prevU = (i == 0) ? DEPOT : routes[r1][i - 1];
                int nextU = (i == routes[r1].size() - 1) ? DEPOT : routes[r1][i + 1];

                // Identify neighbors in Route 2 (for insertion of U after V)
                int nextV = (j == routes[r2].size() - 1) ? DEPOT : routes[r2][j + 1];

                // Calculate Savings from removing U from Route 1
                double removalSavings = vrp.get_dist(prevU, u) + vrp.get_dist(u, nextU) - vrp.get_dist(prevU, nextU);

                // Calculate Cost of inserting U into Route 2 after V
                double insertionCost = vrp.get_dist(v, u) + vrp.get_dist(u, nextV) - vrp.get_dist(v, nextV);

                // Does this move strictly save distance?
                if (insertionCost >= removalSavings - 1e-6)
                    continue;

                // --- APPLY THE MOVE ---
                // 1. Insert U into Route 2 after V
                routes[r2].insert(routes[r2].begin() + j + 1, u);

                // 2. Remove U from Route 1
                routes[r1].erase(routes[r1].begin() + i);

                improvement = true;
                goto restart_search;
            }
        }
    restart_search:;
    }
}

// Writes routes and cost to a .routes file
bool writeRoutes(const std::string &filename, const std::vector<std::vector<int>> &routes, double cost)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }

    // Write each route
    for (size_t i = 0; i < routes.size(); ++i)
    {
        file << "Route #" << (i + 1) << ":";
        for (int node : routes[i])
        {
            file << " " << node;
        }
        file << "\n";
    }

    // Write the cost (using fixed formatting in case it's a floating point number)
    // You can remove std::fixed and std::setprecision if you only want raw integers
    file << "Cost " << std::fixed << std::setprecision(2) << cost << "\n";

    file.close();
    return true;
}

int main(int argc, char *argv[])
{
    try
    {
        // ... ALL of your existing main() code goes here ...
        if (argc < 2)
        {
            std::cerr << "Usage: " << argv[0] << " <filename.vrp>" << std::endl;
            return 1;
        }

        VRP vrp;
        vrp.read(argv[1]);

        // default
        VRP::isRound = false;

        VRP::verbose = false;

        VRP::K = 1500;
        bool print_routes = false;
        // parse arguments
        for (int i = 2; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "-round" && i + 1 < argc)
            {
                VRP::isRound = (std::stoi(argv[i + 1]) == 1);
                i++;
            }
            if (arg == "-verbose")
            {
                VRP::verbose = (std::stoi(argv[i + 1]) == 1);
                ;
                i++;
            }
            if (arg == "-K")
            {
                int temp = std::stoi(argv[i + 1]);
                VRP::K = std::min(temp, (int)vrp.getSize() - 2);
                i++;
            }
            if (arg == "-PR")
            {
                int temp = std::stoi(argv[i + 1]);
                print_routes = (temp == 1);
                i++;
            }
        }

        auto get_neighbour_start = std::chrono::high_resolution_clock::now();
        int num_nodes = vrp.getSize();
        int nbClients = num_nodes - 1; // Exclude depot
        int num_neighbors = vrp.K;

        // 1. Extract coordinates into standard vectors for the KDTree
        std::vector<double> xcoords(num_nodes);
        std::vector<double> ycoords(num_nodes);
        for (int i = 0; i < num_nodes; i++)
        {
            xcoords[i] = vrp.node[i].x; // Assuming vrp.node has .x and .y
            ycoords[i] = vrp.node[i].y;
        }

        std::vector<std::vector<std::pair<weight_t, node_t>>> neighbours(vrp.getSize());
        // 2. Build the KD-Tree (This takes a little time, but saves hours later)
        cobra::KDTree kd_tree(xcoords, ycoords);

        // 3. Calculate savings ONLY for the closest neighbors
        for (int i = 1; i < num_nodes; i++)
        {
            weight_t dist_to_depot = vrp.get_dist(DEPOT, i);

            // Query the KD-tree. We ask for num_neighbors + 1 because the tree usually
            // returns the node 'i' itself as its own closest neighbor.
            std::vector<int> closest_nodes = kd_tree.GetNearestNeighbors(xcoords[i], ycoords[i], num_neighbors + 1);

            for (int j : closest_nodes)
            {
                // Skip if the KD-tree returned the node itself or the depot
                if (i == j || j == DEPOT)
                    continue;

                // Calculate the saving just like your original code
                double saving = dist_to_depot + vrp.get_dist(DEPOT, j) - vrp.get_dist(i, j);

                neighbours[i].push_back({saving, -1 * j});
            }
        }

        auto get_neighbour_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> get_neighbour_time = get_neighbour_end - get_neighbour_start;
        std::cout << "Time taken to compute neighbour list: " << get_neighbour_time.count() << " seconds" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<node_t>> routes = parallel_savings_algorithm(vrp, neighbours);

        if (print_routes)
        {
            std::string filename = argv[1];
            filename = get_base_name(filename) + ".routes";
            weight_t total_cost = calCost(vrp, routes);
            writeRoutes(filename, routes, total_cost);
            return 0; // Exit after writing routes if print_routes flag is set
        }

        std::vector<std::vector<node_t>> clost_nodes(vrp.getSize(), std::vector<node_t>(100));
        for (int i = 1; i < vrp.getSize(); i++)
        {
            for (int j = 0; j < 100; j++)
            {
                clost_nodes[i][j] = -1 * neighbours[i][j].second;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        // if (vrp.verbose)
        // {

        //     std::string filename = argv[1];
        //     filename = get_base_name(filename) + ".edges";
        //     std::ofstream out(filename);
        //     for (auto &p : mergings)
        //     {
        //         for (auto &item : p)
        //             out << item.first << " " << item.second << "\n";
        //         out << "commit\n";
        //     }
        //     out.close();
        // }

        std::chrono::duration<double> elapsed = end_time - start_time;
        weight_t total_cost = calCost(vrp, routes);
        std::cout << "--- Parallel Clarke & Wright Savings Algorithm ---" << std::endl;
        std::cout << "Problem File: " << argv[1] << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Before preprosess Solution Cost: " << total_cost << std::endl;
        xcoords.clear();
        xcoords.shrink_to_fit();

        ycoords.clear();
        ycoords.shrink_to_fit();

        neighbours.clear();
        neighbours.shrink_to_fit();
        auto local_search_start = std::chrono::high_resolution_clock::now();
        routes = postProcessIt(vrp, routes, total_cost);
        // total_cost = calCost(vrp, routes);

        try
        {

            bool globalImprovement = true;
            while (globalImprovement)
            {
                double startCost = calCost(vrp, routes);
                // Write a quick function to sum your cost
                runGranularRelocate(routes, vrp, clost_nodes, vrp.getCapacity(), vrp.getSize() - 1);
                runGranular2OptStar(routes, vrp, clost_nodes, vrp.getCapacity(), vrp.getSize() - 1);

                double endCost = calCost(vrp, routes);

                // Keep looping both operators until NEITHER of them can find an improvement
                globalImprovement = (endCost < startCost - 1e-6);
            }
        }
        catch (const std::string &e)
        {
            std::cerr << "Error in 2-opt*: " << e << std::endl;
        }

        auto local_search_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> local_search_time = local_search_end - local_search_start;

        bool is_valid = verify_sol(vrp, routes, vrp.getCapacity());
        total_cost = calCost(vrp, routes);

        // std::cout << "Threads Used: " << omp_get_max_threads() << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total Solution Cost: " << total_cost << std::endl;
        std::cout << "Number of Routes:   " << routes.size() << std::endl;
        std::cout << "Parrallel Clarke and Wright Time : " << elapsed.count() << std::endl;
        std::cout << "Local Search Time: " << local_search_time.count() << " seconds" << std::endl;
        std::cout << "Total Time Taken:    " << elapsed.count() + local_search_time.count() << " seconds" << std::endl;
        std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID") << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
    }
    catch (const std::string &e)
    {
        // This will catch the hidden error message and print it!
        std::cerr << "CRITICAL ERROR: " << e << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        // Catches standard C++ errors
        std::cerr << "STANDARD ERROR: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        // Catches literally anything else
        std::cerr << "UNKNOWN ERROR OCCURRED!" << std::endl;
        return 1;
    }
    return 0;
}
