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

// CUDA specific headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h> // Required for grid.sync()

#define TWO_PI 6.28318530718
#define PI 3.14159265358979323846
#define PI_VAL 3.14159265359f

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

struct AngularNode
{
    double x;
    double y;
    int id;
    double angle;
};

class VRP
{
public:
    size_t size;
    demand_t capacity;
    std::vector<Point> node;
    std::vector<weight_t> dist_to_depot;
    static double theta;
    static bool isRound;

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
double VRP::theta = 360.00;

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
    if (VRP::isRound)
        return std::round(dist);
    else
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
__device__ volatile unsigned int global_counter_reverse_list = 0;
// __device__ volatile unsigned long long int global_counter_2 = 0;

__global__ void find_buddy_per_node(
    const Point *nodes,
    const node_t *customer_route_map,
    const double *sorted_angles,
    const weight_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    const weight_t *dist_to_depot,
    node_t *crush,
    const demand_t capacity,
    const unsigned int last_index, const double theta_rad)
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
        node_t front_i = route_head[route_id_i];
        node_t back_i = route_tail[route_id_i];
        double my_angle = sorted_angles[i];
        double target_angle = my_angle + theta_rad;
        double wrapped_target_angle = fmod(target_angle, TWO_PI);
        double target_angle_2 = my_angle - theta_rad;
        if (target_angle_2 < 0)
        {
            target_angle_2 += TWO_PI;
        }
        bool in_range = false;
        if (front_i == DEPOT || back_i == DEPOT)
            continue;

        node_t route_id_j, front_j, back_j;
        demand_t tot_demand = route_demands[route_id_i];
        demand_t my_demand = tot_demand;
        double omega;

        for (int j = 1; j <= limit; j++)
        {
            if (j == i)
                continue;
            omega = sorted_angles[j];
            in_range = false;
            if (target_angle_2 <= wrapped_target_angle)
            {
                // Standard case: No wrap-around (e.g., span is 45 to 90 degrees)
                // Omega must be greater than the lower bound AND less than the upper bound.
                if (omega >= target_angle_2 && omega <= wrapped_target_angle)
                {
                    in_range = true;
                }
            }
            else
            {
                // Wrap-around case: Crosses the 0 degree line (e.g., span is 350 to 10 degrees)
                // Omega must be greater than the lower bound OR less than the upper bound.
                if (omega >= target_angle_2 || omega <= wrapped_target_angle)
                {
                    in_range = true;
                }
            }
            if (in_range == false)
                continue;
            route_id_j = customer_route_map[j];
            tot_demand = my_demand + route_demands[route_id_j];
            if (tot_demand > capacity)
                continue;
            front_j = route_head[route_id_j];
            back_j = route_tail[route_id_j];

            if (front_j == DEPOT || back_j == DEPOT)
                continue;

            double saving_1 = dist_to_depot[back_i] + dist_to_depot[front_j] - device_euclidean_dist(nodes[back_i], nodes[front_j]);
            double saving_2 = dist_to_depot[back_j] + dist_to_depot[front_i] - device_euclidean_dist(nodes[back_j], nodes[front_i]);
            double max_sav = max(saving_1, saving_2);
            if (best_saving < max_sav)
            {
                best_saving = max_sav;
                favourite = j;
                final_cap = tot_demand;
            }
            else if (best_saving == max_sav) // Cleaner logic
            {
                if (favourite == -1)
                {
                    favourite = j;
                    continue;
                }
                if (final_cap > tot_demand)
                {
                    favourite = j;
                    final_cap = tot_demand;
                    continue;
                }
                favourite = min(favourite, j);
            }
        }

        crush[i] = favourite;
    }
}

__global__ void get_pairs(
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
    node_t *next_customer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int curr = tid; curr < global_counter; curr += total_threads)
    {
        node_t i = store_i[curr];
        node_t j = store_j[curr];
        node_t route_id_i = customer_route_map[i];
        node_t route_id_j = customer_route_map[j];
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        next_customer[tail_i] = head_j;
        route_tail[route_id_i] = tail_j;
        route_demands[route_id_i] += route_demands[route_id_j];
        route_demands[route_id_j] = 0;
        route_head[route_id_j] = DEPOT;
        route_tail[route_id_j] = DEPOT;
        customer_route_map[j] = -1;
    }
}

__global__ void cleanup(
    node_t *customer_route_map,
    double *sorted_angles,
    node_t *crush, unsigned int last_index,
    unsigned int *slow_pointer)
{
    *slow_pointer = 0;
    global_counter = 0;
    global_counter_reverse_list = 0;
    for (int i = 1; i <= last_index; i++)
    {
        if (customer_route_map[i] != -1)
        {
            (*slow_pointer)++;
            customer_route_map[*slow_pointer] = customer_route_map[i];
            sorted_angles[*slow_pointer] = sorted_angles[i];
        }
        crush[i] = -1;
    }
}

std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp)
{
    const int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    const demand_t CAPACITY = vrp.getCapacity();
    std::cout << "Number of Customers: " << NUM_CUSTOMERS << "\n";

    std::vector<node_t> h_customer_route_map(NUM_CUSTOMERS + 1);
    std::vector<double> h_angles(NUM_CUSTOMERS + 1);
    std::vector<double> h_sorted_angles(NUM_CUSTOMERS + 1);
    std::vector<demand_t> h_route_demands(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_head(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_tail(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_next_customer(vrp.size, DEPOT);
    std::vector<node_t> h_crush(vrp.size, -1);
    unsigned int h_slow_pointer = NUM_CUSTOMERS;

    std::vector<AngularNode> sorted_angular_nodes(NUM_CUSTOMERS + 1);
    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
    {
        double dx = vrp.node[i].x - vrp.node[DEPOT].x;
        double dy = vrp.node[i].y - vrp.node[DEPOT].y;
        double angle = std::atan2(dy, dx);
        if (angle < 0)
            angle += 2 * M_PI; // Normalize to [0, 2PI)
        h_angles[i] = angle;
        sorted_angular_nodes[i] = {vrp.node[i].x, vrp.node[i].y, i, angle};
    }

    // Sort by angle
    std::sort(sorted_angular_nodes.begin() + 1, sorted_angular_nodes.end(),
              [](const AngularNode &a, const AngularNode &b)
              {
                  if (std::abs(a.angle - b.angle) < 1e-6) // Handle doubleing-point precision issues
                  {
                      if (a.x == b.x)
                          return a.y < b.y;
                      return a.x < b.x;
                  }
                  return a.angle < b.angle;
              });

    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
    {
        int id = sorted_angular_nodes[i].id;
        h_customer_route_map[i] = id;
        h_sorted_angles[i] = sorted_angular_nodes[i].angle;
        h_route_demands[i] = vrp.node[id].demand;
        h_route_head[i] = id;
        h_route_tail[i] = id;
    }

    // --- 2. DEVICE: Allocate GPU memory ---
    Point *d_nodes;
    node_t *d_customer_route_map;
    double *d_sorted_angles;
    double *d_angles;
    demand_t *d_route_demands;
    node_t *d_route_head;
    node_t *d_route_tail;
    weight_t *d_dist_to_depot;
    node_t *d_next_customer;
    node_t *d_crush;
    node_t *d_store_i;
    node_t *d_store_j;
    unsigned int *d_slow_pointer;

    dim3 threadsPerBlock(1024);
    dim3 numBlocks((int)(NUM_CUSTOMERS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    // long long totalThreads = threadsPerBlock.x * numBlocks.x;

    checkCudaErrors(cudaMalloc(&d_nodes, (NUM_CUSTOMERS + 1) * sizeof(Point)));
    checkCudaErrors(cudaMalloc(&d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_sorted_angles, (NUM_CUSTOMERS + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_angles, (NUM_CUSTOMERS + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_next_customer, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_crush, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_i, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_j, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_slow_pointer, sizeof(unsigned int)));

    // --- 3. HOST -> DEVICE: Copy data to GPU ---
    checkCudaErrors(cudaMemcpy(d_nodes, vrp.node.data(), (NUM_CUSTOMERS + 1) * sizeof(Point), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_customer_route_map, h_customer_route_map.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sorted_angles, h_sorted_angles.data(), (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_angles, h_angles.data(), (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_head, h_route_head.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_tail, h_route_tail.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_next_customer, h_next_customer.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_crush, h_crush.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slow_pointer, &h_slow_pointer, sizeof(unsigned int), cudaMemcpyHostToDevice));
    double theta_rad = vrp.theta * (PI / 180.0);
    int id = 0;
    unsigned int last_index = NUM_CUSTOMERS;
    std::chrono::time_point<std::chrono::high_resolution_clock> st, en;

    while (true)
    {

        id++;
        if (id == 1)
            st = std::chrono::high_resolution_clock::now();
        /*
        checkCudaErrors(cudaMemcpy(h_sorted_angles.data(), d_sorted_angles, (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_customer_route_map.data(), d_customer_route_map, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
        for (int i = 1; i < last_index; i++)
        {
            if (h_sorted_angles[i] > h_sorted_angles[i + 1])
            {
                std::cout << "Sorted angles are not in order at index: " << i << "\n";
            }
            // std::cout << setprecision(2) << h_sorted_angles[i] << " , ";
            // std::cout << h_customer_route_map[i] << " , ";
        }
        std::cout << "\n";
        std::cout << "last index: " << last_index << "\n";
        */
        find_buddy_per_node<<<numBlocks, threadsPerBlock>>>(
            d_nodes,
            d_customer_route_map,
            d_sorted_angles,
            d_route_demands,
            d_route_head,
            d_route_tail,
            d_dist_to_depot,
            d_crush,
            CAPACITY,
            last_index,
            theta_rad);

        get_pairs<<<numBlocks, threadsPerBlock>>>(
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
            d_next_customer);

        cleanup<<<1, 1>>>(
            d_customer_route_map,
            d_sorted_angles,
            d_crush,
            last_index,
            d_slow_pointer);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(&h_slow_pointer, d_slow_pointer, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // std::cout << h_slow_pointer << " , " << last_index << "\n";
        if (h_slow_pointer == last_index)
        {
            std::cout << "No more positive savings found. Halting." << std::endl;
            std::cout << id << "\n";
            break; // Exit the while loop
        }
        last_index = h_slow_pointer;
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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename.vrp>" << std::endl;
        return 1;
    }

    /*
    CommandLine commandline(argc, argv);
    if (commandline.verbose)
        std::cout << "----- READING INSTANCE: " << commandline.pathInstance << std::endl;

    InstanceCVRPLIB cvrp(commandline.pathInstance, commandline.isRoundingInteger);
    Params params(cvrp.x_coords, cvrp.y_coords, cvrp.service_time, cvrp.demands,
                  cvrp.vehicleCapacity, cvrp.durationLimit, commandline.nbVeh, cvrp.isDurationConstraint, commandline.verbose, commandline.ap);

    */
    VRP vrp;
    vrp.read(argv[1]);

    // default
    VRP::isRound = false;

    // parse arguments
    for (int i = 2; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-round" && i + 1 < argc)
        {
            VRP::isRound = (std::stoi(argv[i + 1]) == 1);
            i++;
        }
        if (arg == "-theta" && i + 1 < argc)
        {
            VRP::theta = std::stod(argv[i + 1]);
            i++;
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<node_t>> routes = parallel_savings_algorithm(vrp);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    weight_t total_cost = calCost(vrp, routes);
    std::cout << "--- Parallel Clarke & Wright Savings Algorithm ---" << std::endl;
    std::cout << "Problem File: " << argv[1] << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Before preprosess Solution Cost: " << total_cost << std::endl;

    auto local_search_start = std::chrono::high_resolution_clock::now();
    routes = postProcessIt(vrp, routes, total_cost);
    total_cost = calCost(vrp, routes);
    /*
    try
    {
        Individual indiv(params, routes);
        LocalSearch localSearch(params);
        localSearch.run(indiv, params.penaltyCapacity, params.penaltyDuration);
        routes = indiv.chromR;
    }
    catch (const std::string &e)
    {
        std::cerr << "HGS Exception: " << e << std::endl;
    }
    */
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

    return 0;
}