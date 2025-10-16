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

// Type definitions consistent with the provided file
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

/**
 * @brief Implements the Seed-Based Clustering heuristic with Farthest Insertion.
 * Clusters customers into routes based on distance and capacity.
 * @param vrp The VRP instance.
 * @return A vector of routes forming a complete solution.
 */
std::vector<std::vector<node_t>> seed_based_clustering(const VRP &vrp)
{
    std::vector<std::vector<node_t>> all_routes;
    std::vector<bool> assigned(vrp.size, false);
    assigned[DEPOT] = true;

    int num_customers_to_assign = vrp.size - 1;
    int num_assigned = 0;

    while (num_assigned < num_customers_to_assign)
    {
        // --- 1. Select a Seed Customer (Farthest from Depot) ---
        node_t seed_node = -1;
        weight_t max_dist = -1.0;
        for (node_t i = 1; i < vrp.size; ++i)
        { // Start from 1 to skip depot
            if (!assigned[i])
            {
                weight_t dist = vrp.get_dist(DEPOT, i);
                if (dist > max_dist)
                {
                    max_dist = dist;
                    seed_node = i;
                }
            }
        }

        if (seed_node == -1)
            break; // All customers assigned

        // --- 2. Start a new cluster (route) with the seed ---
        std::vector<node_t> current_route;
        current_route.push_back(seed_node);
        assigned[seed_node] = true;
        num_assigned++;
        demand_t current_demand = vrp.node[seed_node].demand;

        // --- 3. Grow the Cluster ---
        while (true)
        {
            node_t best_candidate = -1;
            weight_t min_insertion_cost = DBL_MAX;

            // Find the unassigned customer closest to any customer already in the route
            for (node_t u = 1; u < vrp.size; ++u)
            {
                if (!assigned[u] && (current_demand + vrp.node[u].demand <= vrp.getCapacity()))
                {
                    weight_t cost_to_cluster = DBL_MAX;
                    for (node_t c_in_route : current_route)
                    {
                        cost_to_cluster = std::min(cost_to_cluster, vrp.get_dist(u, c_in_route));
                    }

                    if (cost_to_cluster < min_insertion_cost)
                    {
                        min_insertion_cost = cost_to_cluster;
                        best_candidate = u;
                    }
                }
            }

            // If a valid candidate was found, add it to the route
            if (best_candidate != -1)
            {
                current_route.push_back(best_candidate);
                assigned[best_candidate] = true;
                num_assigned++;
                current_demand += vrp.node[best_candidate].demand;
            }
            else
            {
                // No more customers can be added to this route
                break;
            }
        }
        all_routes.push_back(current_route);
    }
    return all_routes;
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
        std::cerr << "Usage: " << argv[0] << " <filename.vrp>" << std::endl;
        return 1;
    }

    // --- Read Problem Data ---
    VRP vrp;
    vrp.read(argv[1]);

    // --- Solve and Time the Algorithm ---
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<node_t>> routes = seed_based_clustering(vrp);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // --- Calculate Cost and Verify Solution ---
    weight_t total_cost = calCost(vrp, routes);
    auto postRoutes = postProcessIt(vrp, routes, total_cost);
    total_cost = calCost(vrp, postRoutes);
    bool is_valid = verify_sol(vrp, postRoutes);

    // --- Print Final Results ---
    std::cout << "--- Seed-Based Clustering (Farthest Insertion) ---" << std::endl;
    std::cout << "Problem File: " << argv[1] << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Solution Cost: " << total_cost << std::endl;
    std::cout << "Total Time Taken:    " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID (Capacity Violated)") << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}
