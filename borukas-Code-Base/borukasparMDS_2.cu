//~~~START:Thu, 16-Jun-2022, 12:43:32 IST
// For GECCO'23 Submission.
// HIGH-PERFORMANCE AND SCALABLE CUDA ACCELERATION
// COMPILE WITH: nvcc -O3 -std=c++14 -gencode arch=compute_75,code=sm_75 borukasparMDS.cu -o borukasparMDS.out
//               (change compute_75 and sm_75 to your GPU's architecture, e.g., sm_86 for Ampere)
/*
 * Original CPU Authors:
 * Rajesh Pandian M | https://mrprajesh.co.in
 * Somesh Singh     | https://ssomesh.github.io
 * Rupesh Nasre     | www.cse.iitm.ac.in/~rupesh
 * N.S.Narayanaswamy| www.cse.iitm.ac.in/~swamy
 * MIT LICENSE
 */

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <deque>
#include <sstream> //stringstream
#include <numeric> // For std::iota
#include <cmath>   // For sqrt

#include <random>
#include <chrono> //timing CPU

// CUDA Includes
#include <cuda_runtime.h>

#define DEBUGCODE 0
#define DEBUG if (DEBUGCODE)

// CUDA Error Checking Macro
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

using namespace std;

//~ Define types
using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;
using ull = unsigned long long;

const node_t DEPOT = 0;

// To store all cmd line params in one struct
class Params
{
public:
    Params()
    {
        toRound = 1;
        nThreads = 20;
    }
    ~Params() {}
    bool toRound;
    short nThreads;
};

class Edge
{
public:
    node_t to;
    weight_t length;

    Edge() {}
    ~Edge() {}
    Edge(node_t t, weight_t l) : to(t), length(l) {}
    bool operator<(const Edge &e) { return length < e.length; }
};

class Point
{
public:
    point_t x;
    point_t y;
    demand_t demand;
};

// To Hold the contents input.vrp
class VRP
{
    demand_t capacity;
    string type;

public:
    size_t size;
    VRP() {}
    ~VRP() {}
    unsigned read(string filename);
    void print();
    weight_t get_dist(node_t i, node_t j, bool isRound = true) const
    {
        if (i == j)
            return 0.0;
        weight_t w = sqrt(pow(node[i].x - node[j].x, 2) + pow(node[i].y - node[j].y, 2));
        if (isRound == false)
            return w;
        return (params.toRound ? round(w) : w);
    }
    vector<Point> node;
    Params params;
    size_t getSize() const { return size; }
    demand_t getCapacity() const { return capacity; }
};

unsigned VRP::read(string filename)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        cerr << "Could not open the file \"" << filename << "\"" << endl;
        exit(1);
    }
    string line;
    for (int i = 0; i < 3; ++i)
        getline(in, line);
    getline(in, line);
    size = stof(line.substr(line.find(":") + 2));
    getline(in, line);
    type = line.substr(line.find(":") + 2);
    getline(in, line);
    capacity = stof(line.substr(line.find(":") + 2));
    getline(in, line);
    node.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        getline(in, line);
        stringstream iss(line);
        size_t id;
        iss >> id >> node[i].x >> node[i].y;
    }
    getline(in, line);
    for (size_t i = 0; i < size; ++i)
    {
        getline(in, line);
        stringstream iss(line);
        size_t id;
        iss >> id >> node[i].demand;
    }
    in.close();
    return capacity;
}

void VRP::print()
{
    cout << "DIMENSION:" << size << '\n';
    cout << "CAPACITY:" << capacity << '\n';
    for (auto i = 0u; i < size; ++i)
    {
        cout << i << ':' << setw(6) << node[i].x << ' '
             << setw(6) << node[i].y << ' '
             << setw(6) << node[i].demand << endl;
    }
}

// START: CUDA BORUVKA'S ALGORITHM IMPLEMENTATION
struct MSTEdge
{
    node_t u, v;
};

__device__ int find_repres(volatile int *comp, int v)
{
    int p = comp[v];
    if (v == p)
        return v;
    int gp = comp[p];
    while (p != gp)
    {
        atomicCAS((int *)&comp[v], p, gp);
        v = gp;
        p = comp[v];
        gp = comp[p];
    }
    return p;
}

__global__ void initialize_kernel(int V, double *d_cheapest_weight)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V)
    {
        d_cheapest_weight[tid] = DBL_MAX;
    }
}

__global__ void find_cheapest_kernel(int V, const point_t *d_xs, const point_t *d_ys, volatile int *d_comp,
                                     double *d_cheapest_weight, int *d_cheapest_u, int *d_cheapest_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    for (int u_node = i; u_node < V; u_node += gridSize)
    {
        int u_rep = find_repres(d_comp, u_node);
        for (int v_node = u_node + 1; v_node < V; ++v_node)
        {
            int v_rep = find_repres(d_comp, v_node);
            if (u_rep != v_rep)
            {
                weight_t w = sqrt(pow(d_xs[u_node] - d_xs[v_node], 2) + pow(d_ys[u_node] - d_ys[v_node], 2));

                // Atomically update cheapest edge for component u_rep using CAS loop
                double old_w = d_cheapest_weight[u_rep];
                while (w < old_w)
                {
                    ull old_w_ull = __double_as_longlong(old_w);
                    ull new_w_ull = __double_as_longlong(w);
                    ull result = atomicCAS((ull *)&d_cheapest_weight[u_rep], old_w_ull, new_w_ull);
                    if (result == old_w_ull)
                    { // If swap was successful
                        d_cheapest_u[u_rep] = u_node;
                        d_cheapest_v[u_rep] = v_node;
                        break;
                    }
                    old_w = __longlong_as_double(result); // If failed, retry with the newer value
                }

                // Atomically update cheapest edge for component v_rep using CAS loop
                old_w = d_cheapest_weight[v_rep];
                while (w < old_w)
                {
                    ull old_w_ull = __double_as_longlong(old_w);
                    ull new_w_ull = __double_as_longlong(w);
                    ull result = atomicCAS((ull *)&d_cheapest_weight[v_rep], old_w_ull, new_w_ull);
                    if (result == old_w_ull)
                    {
                        d_cheapest_u[v_rep] = u_node;
                        d_cheapest_v[v_rep] = v_node;
                        break;
                    }
                    old_w = __longlong_as_double(result);
                }
            }
        }
    }
}

__global__ void merge_kernel(int V, volatile int *d_comp,
                             double *d_cheapest_weight, int *d_cheapest_u, int *d_cheapest_v,
                             MSTEdge *d_mst_edges, int *d_mst_count, volatile bool *d_active)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    for (int i = tid; i < V; i += gridSize)
    {
        if (d_comp[i] != i)
            continue; // Process only root nodes

        if (d_cheapest_weight[i] != DBL_MAX)
        {
            int u = d_cheapest_u[i];
            int v = d_cheapest_v[i];

            int u_rep = find_repres(d_comp, u);
            int v_rep = find_repres(d_comp, v);

            if (u_rep != v_rep)
            {
                int high_rep = max(u_rep, v_rep);
                int low_rep = min(u_rep, v_rep);
                int old_root = atomicCAS((int *)&d_comp[low_rep], low_rep, high_rep);

                if (old_root == low_rep)
                {
                    int mst_idx = atomicAdd(d_mst_count, 1);
                    d_mst_edges[mst_idx] = {u, v};
                    *d_active = true;
                }
            }
        }
    }
}

// Host function to orchestrate the CUDA Boruvka's MST
std::vector<std::vector<Edge>> BoruvkasAlgoCUDA(const VRP &vrp)
{
    int N = vrp.getSize();
    if (N <= 1)
        return std::vector<std::vector<Edge>>(N);

    // 1. Create coordinate lists on the host
    std::vector<point_t> h_xs(N), h_ys(N);
    for (int i = 0; i < N; ++i)
    {
        h_xs[i] = vrp.node[i].x;
        h_ys[i] = vrp.node[i].y;
    }

    // 2. Allocate GPU memory
    point_t *d_xs, *d_ys;
    MSTEdge *d_mst_edges;
    int *d_comp, *d_mst_count;
    double *d_cheapest_weight;
    int *d_cheapest_u, *d_cheapest_v;
    bool *d_active;

    gpuErrchk(cudaMalloc(&d_xs, N * sizeof(point_t)));
    gpuErrchk(cudaMalloc(&d_ys, N * sizeof(point_t)));
    gpuErrchk(cudaMalloc(&d_mst_edges, (N - 1) * sizeof(MSTEdge)));
    gpuErrchk(cudaMalloc(&d_comp, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_cheapest_weight, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_cheapest_u, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_cheapest_v, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_mst_count, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_active, sizeof(bool)));

    // 3. Initialize and transfer data to GPU
    std::vector<int> h_comp(N);
    std::iota(h_comp.begin(), h_comp.end(), 0);

    gpuErrchk(cudaMemcpy(d_xs, h_xs.data(), N * sizeof(point_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_ys, h_ys.data(), N * sizeof(point_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_comp, h_comp.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_mst_count, 0, sizeof(int)));

    // 4. Main loop on host, launching standard kernels
    bool h_active = true;
    while (h_active)
    {
        h_active = false;
        cudaMemcpy(d_active, &h_active, sizeof(bool), cudaMemcpyHostToDevice);

        int threadsPerBlock = 1024;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        initialize_kernel<<<blocks, threadsPerBlock>>>(N, d_cheapest_weight);
        // gpuErrchk(cudaGetLastError());

        find_cheapest_kernel<<<blocks, threadsPerBlock>>>(N, d_xs, d_ys, d_comp, d_cheapest_weight, d_cheapest_u, d_cheapest_v);
        // gpuErrchk(cudaGetLastError());

        merge_kernel<<<blocks, threadsPerBlock>>>(N, d_comp, d_cheapest_weight, d_cheapest_u, d_cheapest_v, d_mst_edges, d_mst_count, d_active);
        // gpuErrchk(cudaGetLastError());

        cudaMemcpy(&h_active, d_active, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    // 5. Copy result back
    int h_mst_count;
    gpuErrchk(cudaMemcpy(&h_mst_count, d_mst_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<MSTEdge> h_mst_edges(h_mst_count);
    if (h_mst_count > 0)
    {
        gpuErrchk(cudaMemcpy(h_mst_edges.data(), d_mst_edges, h_mst_count * sizeof(MSTEdge), cudaMemcpyDeviceToHost));
    }

    // 6. Free memory
    gpuErrchk(cudaFree(d_xs));
    gpuErrchk(cudaFree(d_ys));
    gpuErrchk(cudaFree(d_mst_edges));
    gpuErrchk(cudaFree(d_comp));
    gpuErrchk(cudaFree(d_cheapest_weight));
    gpuErrchk(cudaFree(d_cheapest_u));
    gpuErrchk(cudaFree(d_cheapest_v));
    gpuErrchk(cudaFree(d_mst_count));
    gpuErrchk(cudaFree(d_active));

    // 7. Convert to adjacency list
    std::vector<std::vector<Edge>> nG(N);
    for (const auto &edge : h_mst_edges)
    {
        weight_t w = vrp.get_dist(edge.u, edge.v, vrp.params.toRound);
        nG[edge.u].push_back(Edge(edge.v, w));
        nG[edge.v].push_back(Edge(edge.u, w));
    }
    return nG;
}

// END: CUDA BORUVKA'S ALGORITHM IMPLEMENTATION

// THE REST OF THE VRP SOLVER CODE IS UNCHANGED
void ShortCircutTour(std::vector<std::vector<Edge>> &g, std::vector<bool> &visited, node_t u, std::vector<node_t> &out)
{
    visited[u] = true;
    out.push_back(u);
    for (auto e : g[u])
    {
        if (!visited[e.to])
        {
            ShortCircutTour(g, visited, e.to, out);
        }
    }
}

std::vector<std::vector<node_t>>
Split_convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute)
{
    std::vector<node_t> customer_tour;
    customer_tour.reserve(vrp.size);
    for (node_t node : singleRoute)
    {
        if (node != DEPOT)
        {
            customer_tour.push_back(node);
        }
    }
    const int n = customer_tour.size();
    if (n == 0)
        return {};
    std::vector<double> sum_demands(n + 1, 0.0);
    std::vector<double> sum_dist(n + 1, 0.0);
    for (int i = 0; i < n; ++i)
    {
        sum_demands[i + 1] = sum_demands[i] + vrp.node[customer_tour[i]].demand;
        if (i > 0)
        {
            sum_dist[i + 1] = sum_dist[i] + vrp.get_dist(customer_tour[i - 1], customer_tour[i]);
        }
    }
    std::vector<weight_t> V(n + 1, std::numeric_limits<weight_t>::max());
    std::vector<int> P(n + 1, -1);
    V[0] = 0;
    std::deque<int> q;
    q.push_back(0);
    for (int j = 1; j <= n; ++j)
    {
        while (!q.empty() && sum_demands[j] - sum_demands[q.front()] > vrp.getCapacity())
        {
            q.pop_front();
        }
        auto calculate_total_cost = [&](int i)
        {
            double route_dist;
            if (i == j - 1)
            {
                route_dist = vrp.get_dist(DEPOT, customer_tour[i]) + vrp.get_dist(customer_tour[i], DEPOT);
            }
            else
            {
                route_dist = vrp.get_dist(DEPOT, customer_tour[i]) + (sum_dist[j] - sum_dist[i + 1]) + vrp.get_dist(customer_tour[j - 1], DEPOT);
            }
            return V[i] + route_dist;
        };
        while (q.size() >= 2)
        {
            if (calculate_total_cost(q[0]) >= calculate_total_cost(q[1]))
            {
                q.pop_front();
            }
            else
            {
                break;
            }
        }
        if (!q.empty())
        {
            P[j] = q.front();
            V[j] = calculate_total_cost(P[j]);
        }
        auto g = [&](int i)
        {
            if (i == 0)
                return 0.0;
            return V[i] - sum_dist[i] + vrp.get_dist(DEPOT, customer_tour[i - 1]);
        };
        while (!q.empty() && g(q.back()) >= g(j))
        {
            q.pop_back();
        }
        q.push_back(j);
    }
    std::vector<std::vector<node_t>> final_routes;
    int current_idx = n;
    while (current_idx > 0)
    {
        int pred_idx = P[current_idx];
        std::vector<node_t> new_route;
        for (int k = pred_idx; k < current_idx; ++k)
        {
            new_route.push_back(customer_tour[k]);
        }
        final_routes.push_back(new_route);
        current_idx = pred_idx;
    }
    std::reverse(final_routes.begin(), final_routes.end());
    return final_routes;
}

std::pair<weight_t, std::vector<std::vector<node_t>>>
calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes)
{
    weight_t total_cost = 0.0;
    for (const auto &route : final_routes)
    {
        if (route.empty())
            continue;
        weight_t curr_route_cost = 0;
        curr_route_cost += vrp.get_dist(DEPOT, route[0]);
        for (unsigned jj = 1; jj < route.size(); ++jj)
        {
            curr_route_cost += vrp.get_dist(route[jj - 1], route[jj]);
        }
        curr_route_cost += vrp.get_dist(route.back(), DEPOT);
        total_cost += curr_route_cost;
    }
    return {total_cost, final_routes};
}

void tsp_2opt(const VRP &vrp, std::vector<node_t> &cities)
{
    if (cities.size() < 2)
        return;
    bool improved = true;
    while (improved)
    {
        improved = false;
        weight_t best_distance = calCost(vrp, {cities}).first;

        for (size_t i = 0; i < cities.size() - 1; i++)
        {
            for (size_t k = i + 1; k < cities.size(); k++)
            {
                std::vector<node_t> new_route = cities;
                std::reverse(new_route.begin() + i, new_route.begin() + k + 1);
                weight_t new_distance = calCost(vrp, {new_route}).first;
                if (new_distance < best_distance)
                {
                    cities = new_route;
                    best_distance = new_distance;
                    improved = true;
                }
            }
        }
    }
}

std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t &minCost)
{
    std::vector<std::vector<node_t>> postprocessed_routes = final_routes;
#pragma omp parallel for
    for (unsigned i = 0; i < postprocessed_routes.size(); ++i)
    {
        tsp_2opt(vrp, postprocessed_routes[i]);
    }
    minCost = calCost(vrp, postprocessed_routes).first;
    return postprocessed_routes;
}

bool verify_sol(const VRP &vrp, const vector<vector<node_t>> &final_routes, unsigned capacity)
{
    std::vector<bool> visited(vrp.getSize(), false);
    visited[DEPOT] = true;
    for (const auto &route : final_routes)
    {
        demand_t route_demand = 0;
        for (node_t node : route)
        {
            if (visited[node])
                return false; // Visited twice
            visited[node] = true;
            route_demand += vrp.node[node].demand;
        }
        if (route_demand > capacity)
            return false; // Exceeds capacity
    }
    for (size_t i = 1; i < vrp.getSize(); ++i)
    {
        if (!visited[i])
            return false; // Not all customers visited
    }
    return true;
}

// MAIN
int main(int argc, char *argv[])
{
    VRP vrp;
    if (argc < 2)
    {
        cout << "parMDS version 1.2 (CUDA Boruvka's MST)" << '\n';
        cout << "Usage: " << argv[0] << " toy.vrp [-nthreads <n>] [-round 0|1]" << '\n';
        exit(1);
    }
    for (int ii = 2; ii < argc; ii += 2)
    {
        if (string(argv[ii]) == "-round")
            vrp.params.toRound = atoi(argv[ii + 1]);
        else if (string(argv[ii]) == "-nthreads")
            vrp.params.nThreads = atoi(argv[ii + 1]);
    }
    vrp.read(argv[1]);

    auto start_total = chrono::high_resolution_clock::now();

    // =========================================================================
    // MODIFICATION: Call CUDA Boruvka's Algorithm
    // =========================================================================
    auto start_mst = chrono::high_resolution_clock::now();
    auto mstG = BoruvkasAlgoCUDA(vrp);
    auto end_mst = chrono::high_resolution_clock::now();
    // =========================================================================

    auto start_refine = chrono::high_resolution_clock::now();
    std::vector<node_t> singleRoute;
    std::vector<bool> visited(mstG.size(), false);
    ShortCircutTour(mstG, visited, 0, singleRoute);

    auto initial_routes = Split_convertToVrpRoutes(vrp, singleRoute);
    auto cost_pair = calCost(vrp, initial_routes);
    auto end_refine = chrono::high_resolution_clock::now();

    auto start_post = chrono::high_resolution_clock::now();
    weight_t finalCost = cost_pair.first;
    auto postRoutes = postProcessIt(vrp, cost_pair.second, finalCost);
    auto end_post = chrono::high_resolution_clock::now();

    auto end_total = chrono::high_resolution_clock::now();

    chrono::duration<double> mst_duration = end_mst - start_mst;
    chrono::duration<double> refine_duration = end_refine - start_refine;
    chrono::duration<double> post_duration = end_post - start_post;
    chrono::duration<double> total_duration = end_total - start_total;

    bool verified = verify_sol(vrp, postRoutes, vrp.getCapacity());

    cout << argv[1] << " ";
    cout << "Initial Cost = " << fixed << setprecision(2) << cost_pair.first << ',';
    cout << "Final Cost = " << finalCost;
    cout << " | Time(s): ";
    cout << "MST(CUDA) = " << fixed << setprecision(4) << mst_duration.count() << ',';
    cout << "Refinement = " << refine_duration.count() << ',';
    cout << "Post-Proc = " << post_duration.count() << ',';
    cout << "Total = " << total_duration.count();
    cout << (verified ? " VALID" : " INVALID") << endl;

    return 0;
}
