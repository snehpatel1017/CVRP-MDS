//~~~START:Thu, 16-Jun-2022, 12:43:32 IST
// For GECCO'23 Submission.
// nvc++ -O3 -std=c++14 -acc=multicore  parMDS.cpp -o parMDS.out && time ./parMDS.out toy.vrp 32
/*
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
#include "delaunator.hpp"
#include <random>
#include <chrono> //timing CPU

unsigned DEBUGCODE = 0;
#define DEBUG if (DEBUGCODE)

using namespace std;

//~ Define types
using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int; // let's keep as int than unsigned. -1 is init. nodes ids 0 to n-1

const node_t DEPOT = 0; // CVRP depot is always assumed to be zero.

// To store all cmd line params in one struct
class Params
{
public:
    Params()
    {
        toRound = 1;   // DEFAULT is round
        nThreads = 20; // DEFAULT is 20 OMP threads
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
    Edge(node_t t, weight_t l)
    {
        to = t;
        length = l;
    }
    bool operator<(const Edge &e)
    {
        return length < e.length;
    }
};

class Point
{
public:
    //~ int id; // may be needed later for SS.
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

    void print_dist();

    std::vector<std::vector<Edge>> cal_graph_dist();
    weight_t get_dist(node_t i, node_t j, bool isRound = true) const
    {
        if (i == j)
            return 0.0;
        node_t temp;
        if (i > j)
        {
            temp = i;
            i = j;
            j = temp;
        }
        weight_t w = sqrt(((node[i].x - node[j].x) * (node[i].x - node[j].x)) + ((node[i].y - node[j].y) * (node[i].y - node[j].y)));
        if (isRound == false)
            return w;
        return (params.toRound ? round(w) : w);
    }

public:
    vector<Point> node;

    Params params;

    size_t getSize() const
    {
        return size;
    }
    demand_t getCapacity() const
    {
        return capacity;
    }
};

// Prints distance of every pair of nodes
void VRP::print_dist()
{
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << i << ":";
        for (size_t j = 0; j < size; ++j)
        {
            cout << setw(10) << get_dist(i, j) << ' ';
        }
        std::cout << std::endl;
    }
}

// Parsing/Reading the .vrp file!
unsigned VRP::read(string filename)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        std::cerr << "Could not open the file \"" << filename << "\"" << std::endl;
        exit(1);
    }
    string line;
    for (int i = 0; i < 3; ++i)
        getline(in, line);

    // DIMENSION
    getline(in, line);
    size = stof(line.substr(line.find(":") + 2));
    // cout << "MYSIZE = " << size << endl;

    // DISTANCE TYPE
    getline(in, line);
    type = line.find(":");

    // CAPACITY
    getline(in, line);
    capacity = stof(line.substr(line.find(":") + 2));

    // skip NODE_COORD_SECTION
    getline(in, line);

    // Allocate
    node.resize(size);

    for (size_t i = 0; i < size; ++i)
    {
        getline(in, line);

        stringstream iss(line);
        size_t id;
        string xStr, yStr;

        iss >> id >> xStr >> yStr;
        node[i].x = stof(xStr);
        node[i].y = stof(yStr);
    }

    // skip DEMAND_SECTION
    getline(in, line);

    for (size_t i = 0; i < size; ++i)
    {
        getline(in, line);
        stringstream iss(line);
        size_t id;
        string dStr;

        iss >> id >> dStr;

        node[i].demand = stof(dStr);
    }
    in.close();

    return capacity;
}

// To print and check if read it okay.
void VRP::print()
{
    std::cout << "DIMENSION:" << size << '\n';
    std::cout << "CAPACITY:" << capacity << '\n';
    for (auto i = 0u; i < size; ++i)
    {
        std::cout << i << ':'
                  << setw(6) << node[i].x << ' '
                  << setw(6) << node[i].y << ' '
                  << setw(6) << node[i].demand << std::endl;
    }
}

// Graph's Adjacency information.
void printAdjList(const std::vector<std::vector<Edge>> &graph)
{
    int i = 0;
    for (auto vec : graph)
    {
        std::cout << i << ": ";
        for (auto e : vec)
        {
            std::cout << e.to << " ";
        }
        i++;
        std::cout << std::endl;
    }
}

// DFS Recursive.
void ShortCircutTour(std::vector<std::vector<Edge>> &g, std::vector<bool> &visited, node_t u, std::vector<node_t> &out)
{
    visited[u] = true;
    DEBUG std::cout << u << ' ';
    out.push_back(u);
    for (auto e : g[u])
    {
        node_t v = e.to;
        if (!visited[v])
        {
            ShortCircutTour(g, visited, v, out);
        }
    }
}

// Converts a permutation to set of routes
std::vector<std::vector<node_t>>
convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute)
{
    std::vector<std::vector<node_t>> routes;

    demand_t vCapacity = vrp.getCapacity();
    demand_t residueCap = vCapacity;
    std::vector<node_t> aRoute;

    for (auto v : singleRoute)
    {
        if (v == 0)
            continue;
        if (residueCap - vrp.node[v].demand >= 0)
        {
            aRoute.push_back(v);
            residueCap = residueCap - vrp.node[v].demand;
        }
        else
        { // new route
            routes.push_back(aRoute);
            aRoute.clear();
            aRoute.push_back(v);
            residueCap = vCapacity - vrp.node[v].demand;
        }
    }
    routes.push_back(aRoute);
    return routes;
}

weight_t calRouteValue(const VRP &vrp, const std::vector<node_t> &aRoute, node_t depot = 1)
{
    weight_t routeVal = 0;
    node_t prevPoint = 0;

    for (auto aPoint : aRoute)
    {
        routeVal += vrp.get_dist(prevPoint, aPoint);
        prevPoint = aPoint;
    }
    routeVal += vrp.get_dist(prevPoint, 0);

    return routeVal;
}

void printOutput(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes)
{
    weight_t total_cost = 0.0;

    for (unsigned ii = 0; ii < final_routes.size(); ++ii)
    {
        std::cout << "Route #" << ii + 1 << ":";
        for (unsigned jj = 0; jj < final_routes[ii].size(); ++jj)
        {
            std::cout << " " << final_routes[ii][jj];
        }
        std::cout << '\n';
    }

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

    std::cout << "Cost " << total_cost << std::endl;
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

std::pair<weight_t, std::vector<std::vector<node_t>>>
calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes)
{
    weight_t total_cost = 0.0;

#pragma omp parallel for reduction(+ : total_cost)
    for (unsigned ii = 0; ii < final_routes.size(); ++ii)
    {
        weight_t curr_route_cost = 0;
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);

#pragma omp parallel for reduction(+ : curr_route_cost)
        for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
        {
            curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
        }
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);
        total_cost += curr_route_cost;
    }
    return {total_cost, final_routes};
}

bool verify_sol(const VRP &vrp, vector<vector<node_t>> final_routes, unsigned capacity)
{
    unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
    memset(hist, 0, sizeof(unsigned) * vrp.getSize());

    for (unsigned i = 0; i < final_routes.size(); ++i)
    {
        unsigned route_sum_of_demands = 0;
        for (unsigned j = 0; j < final_routes[i].size(); ++j)
        {
            route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
            hist[final_routes[i][j]] += 1;
        }
        if (route_sum_of_demands > capacity)
        {
            free(hist);
            return false;
        }
    }

    for (unsigned i = 1; i < vrp.getSize(); ++i)
    {
        if (hist[i] > 1)
        {
            free(hist);
            return false;
        }
        if (hist[i] == 0)
        {
            free(hist);
            return false;
        }
    }
    free(hist);
    return true;
}

struct DSU_for_Kruskal
{
    std::vector<node_t> parent;
    DSU_for_Kruskal(size_t n)
    {
        parent.resize(n);
        std::iota(parent.begin(), parent.end(), 0);
    }
    node_t find(node_t i)
    {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }
    void unite(node_t i, node_t j)
    {
        node_t root_i = find(i);
        node_t root_j = find(j);
        if (root_i != root_j)
            parent[root_i] = root_j;
    }
};

struct GraphEdge
{
    node_t u, v;
    weight_t weight;
    bool operator<(const GraphEdge &other) const
    {
        return weight < other.weight;
    }
};

// O(N log N) EMST algorithm using Delaunay Triangulation + Kruskal's
std::vector<std::vector<Edge>>
EMST_Delaunay_Kruskal(const VRP &vrp)
{
    auto N = vrp.getSize();
    if (N == 0)
        return {};

    // 1. Prepare coordinates for the delaunator library
    std::vector<double> coords;
    coords.reserve(N * 2);
    for (const auto &p : vrp.node)
    {
        coords.push_back(p.x);
        coords.push_back(p.y);
    }

    // 2. Compute the Delaunay Triangulation (O(N log N))
    delaunator::Delaunator d(coords);

    // 3. Build the edge list from the triangulation
    std::vector<GraphEdge> graph_edges;
    for (std::size_t i = 0; i < d.triangles.size(); i += 3)
    {
        node_t p1 = d.triangles[i];
        node_t p2 = d.triangles[i + 1];
        node_t p3 = d.triangles[i + 2];

        graph_edges.push_back({p1, p2, vrp.get_dist(p1, p2, false)});
        graph_edges.push_back({p2, p3, vrp.get_dist(p2, p3, false)});
        graph_edges.push_back({p3, p1, vrp.get_dist(p3, p1, false)});
    }

    // =========================================================================
    // START: FIX
    // 3.5. Guarantee graph connectivity. The triangulation may be disconnected
    // for some degenerate inputs. To ensure Kruskal's finds a full spanning
    // tree, we add edges from the depot to all other nodes. Kruskal's will
    // only use these edges if necessary to connect components.
    for (node_t i = 1; i < N; ++i)
    {
        graph_edges.push_back({DEPOT, i, vrp.get_dist(DEPOT, i, false)});
    }
    // END: FIX
    // =========================================================================

    // 4. Run Kruskal's algorithm on the (now guaranteed connected) graph
    std::sort(graph_edges.begin(), graph_edges.end());

    DSU_for_Kruskal dsu(N);
    std::vector<std::vector<Edge>> nG(N);
    int edges_in_mst = 0;

    for (const auto &edge : graph_edges)
    {
        if (dsu.find(edge.u) != dsu.find(edge.v))
        {
            dsu.unite(edge.u, edge.v);
            nG[edge.u].push_back(Edge(edge.v, edge.weight));
            nG[edge.v].push_back(Edge(edge.u, edge.weight));
            edges_in_mst++;
            if (edges_in_mst == N - 1)
                break;
        }
    }

    return nG;
}

int main(int argc, char *argv[])
{
    VRP vrp;
    if (argc < 2)
    {
        std::cout << "parMDS version 1.1 (Boruvka's MST)" << '\n';
        std::cout << "Usage: " << argv[0] << " toy.vrp [-nthreads <n> DEFAULT is 20] [-round 0 or 1 DEFAULT:1]" << '\n';
        exit(1);
    }

    for (int ii = 2; ii < argc; ii += 2)
    {
        if (std::string(argv[ii]) == "-round")
            vrp.params.toRound = atoi(argv[ii + 1]);
        else if (std::string(argv[ii]) == "-nthreads")
            vrp.params.nThreads = atoi(argv[ii + 1]);
        else
        {
            std::cerr << "INVALID Arguments!" << '\n';
            std::cerr << "Usage:" << argv[0] << " toy.vrp -nthreads 20 -round 1" << '\n';
            exit(1);
        }
    }

    vrp.read(argv[1]);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // MODIFICATION: Replace Prim's with Boruvka's Algorithm
    // =========================================================================
    auto mstG = EMST_Delaunay_Kruskal(vrp);
    // =========================================================================

    std::vector<bool> visited(mstG.size(), false);
    visited[0] = true;
    std::vector<int> singleRoute;

    weight_t minCost = DBL_MAX;
    std::vector<std::vector<node_t>> minRoute;

    auto mstCopy = mstG;

    for (int i = 0; i < 1; i++)
    {
        for (auto &list : mstCopy)
        {
            std::shuffle(list.begin(), list.end(), std::default_random_engine(0));
        }

        std::vector<int> singleRoute;

        std::vector<bool> visited(mstCopy.size(), false);
        visited[0] = true;

        ShortCircutTour(mstCopy, visited, 0, singleRoute);
        DEBUG std::cout << '\n';

        auto aRoutes = convertToVrpRoutes(vrp, singleRoute);

        auto aCostRoute = calCost(vrp, aRoutes);
        if (aCostRoute.first < minCost)
        {
            minCost = aCostRoute.first;
            minRoute = aCostRoute.second;
        }
    }

    auto minCost1 = minCost;

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    uint64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    auto timeUpto1 = (double)(elapsed * 1.E-9);
    short PARLIMIT = vrp.params.nThreads;
    std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
    std::vector<double> Middle_Part_Times(4, 0);

#pragma omp parallel for shared(minCost, minRoute) num_threads(PARLIMIT)
    for (int i = 0; i < 100000; i += PARLIMIT)
    {
        std::chrono::high_resolution_clock::time_point st = std::chrono::high_resolution_clock::now();
        for (auto &list : mstCopy)
        {
            std::shuffle(list.begin(), list.end(), std::default_random_engine(rand()));
        }
        std::chrono::high_resolution_clock::time_point en = std::chrono::high_resolution_clock::now();
        uint64_t elp = std::chrono::duration_cast<std::chrono::nanoseconds>(en - st).count();
        Middle_Part_Times[0] += (double)(elp * 1.E-9);

        std::vector<int> singleRoute;
        std::vector<bool> visited(mstCopy.size(), false);
        visited[0] = true;

        st = std::chrono::high_resolution_clock::now();
        ShortCircutTour(mstCopy, visited, 0, singleRoute);
        DEBUG std::cout << '\n';
        en = std::chrono::high_resolution_clock::now();
        elp = std::chrono::duration_cast<std::chrono::nanoseconds>(en - st).count();
        Middle_Part_Times[1] += (double)(elp * 1.E-9);

        st = std::chrono::high_resolution_clock::now();
        auto aRoutes = convertToVrpRoutes(vrp, singleRoute);
        en = std::chrono::high_resolution_clock::now();
        elp = std::chrono::duration_cast<std::chrono::nanoseconds>(en - st).count();
        Middle_Part_Times[2] += (double)(elp * 1.E-9);

        st = std::chrono::high_resolution_clock::now();
        auto aCostRoute = calCost(vrp, aRoutes);
        en = std::chrono::high_resolution_clock::now();
        elp = std::chrono::duration_cast<std::chrono::nanoseconds>(en - st).count();
        Middle_Part_Times[3] += (double)(elp * 1.E-9);
        if (aCostRoute.first < minCost)
        {
            minCost = aCostRoute.first;
            minRoute = aCostRoute.second;
        }
    }

    auto minCost2 = minCost;
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start2).count();

    auto timeUpto2 = (double)(elapsed * 1.E-9);
    std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

    auto postRoutes = postProcessIt(vrp, minRoute, minCost);

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start3).count();
    double timeUpto3 = (double)(elapsed * 1.E-9);
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double total_time = (double)(elapsed * 1.E-9);

    bool verified = false;
    verified = verify_sol(vrp, postRoutes, vrp.getCapacity());

    std::cout << argv[1] << " Cost ";
    // std::cout << minCost1 << ' ';
    std::cout << "Pre-Refine COST = ";
    std::cout << minCost2 << '\n';
    std::cout << "Pre-Processed COST = ";
    std::cout << minCost << "\n";

    // Execution time after Step 1, Step 2 & 3, and Step 4.
    std::cout << " Time(seconds) ";
    std::cout << "Time for MST = ";
    std::cout << timeUpto1 << '\n';
    std::cout << "Time for Middle part = ";
    std::cout << timeUpto2 << '\n';
    std::cout << "Time for Preprocessing = ";
    std::cout << timeUpto3 << "\n";
    std::cout << "total time = ";
    std::cout << total_time << "\n";

    // middle part time breakdowns
    std::cout << "Time for shuffle = ";
    std::cout << Middle_Part_Times[0] << "\n";
    std::cout << "Time for ShortCircutTour = ";
    std::cout << Middle_Part_Times[1] << "\n";
    std::cout << "Time for Split_convertToVrpRoutes = ";
    std::cout << Middle_Part_Times[2] << "\n";
    std::cout << "Time for calCost = ";
    std::cout << Middle_Part_Times[3] << "\n";

    if (verified)
        std::cout << " VALID" << std::endl;
    else
        std::cout << " INVALID" << std::endl;

    return 0;
}