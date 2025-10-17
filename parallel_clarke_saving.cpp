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
#include <omp.h>   // For OpenMP
#include <cstring>

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

// Data structure to hold a calculated saving
struct Saving
{
    node_t i, j;
    weight_t value;

    // For sorting in descending order
    bool operator<(const Saving &other) const
    {
        // Primary sort key: higher saving value comes first.
        if (value > other.value)
            return true;
        if (value < other.value)
            return false;

        // --- Tie-breaking logic ---
        // If saving values are identical, sort by customer indices to ensure
        // a consistent, deterministic order every time.

        // To handle pairs like (5, 10) and (10, 5) as identical, we
        // create canonical pairs of {min_id, max_id}.
        auto p1 = std::minmax(i, j);
        auto p2 = std::minmax(other.i, other.j);

        // Secondary sort key: the smaller customer ID in the pair.
        if (p1.first < p2.first)
            return true;
        if (p1.first > p2.first)
            return false;

        // Tertiary sort key: the larger customer ID in the pair.
        if (p1.second < p2.second)
            return true;

        return false;
    }
};
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
bool verify_sol(const VRP &vrp, vector<vector<node_t>> final_routes, unsigned capacity)
{
    /* verifies if the solution is valid or not */
    /**
     * 1. All vertices appear in the solution exactly once.
     * 2. For every route, the capacity constraint is respected.
     **/

    // unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
    // memset(hist, 0, sizeof(unsigned) * vrp.getSize());
    std::vector<unsigned int> hist(vrp.getSize(), 0);

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
            std::cout << i << " jaju\n";
            return false;
        }
        if (hist[i] == 0)
        {
            std::cout << i << " missing\n";
            return false;
        }
    }
    return true;
}

/**
 * @brief Implements the massively parallel Clarke and Wright Savings algorithm.
 * @param vrp The VRP instance.
 * @return A vector of routes forming a complete solution.
 */
std::vector<std::vector<node_t>> parallel_savings_algorithm(const VRP &vrp)
{
    // --- 1. Calculate Savings (Extremely Parallel Part) ---
    std::vector<Saving> savings_list;
    // Reserve memory to avoid reallocations within the parallel block
    savings_list.reserve((vrp.size * (vrp.size - 1)) / 2);

#pragma omp parallel for collapse(2)
    for (node_t i = 1; i < vrp.size; ++i)
    {
        for (node_t j = i + 1; j < vrp.size; ++j)
        {
            weight_t saving_value = vrp.dist_to_depot[i] + vrp.dist_to_depot[j] - vrp.get_dist(i, j);
            if (saving_value > 0)
            {
#pragma omp critical
                savings_list.push_back({i, j, saving_value});
            }
        }
    }

    // --- 2. Sort Savings (Sequential but fast) ---
    std::sort(savings_list.begin(), savings_list.end());
    std::vector<demand_t> route_demands(vrp.size);
    std::vector<node_t> customer_route_map(vrp.size);
    std::vector<node_t> route_head(vrp.size);
    std::vector<node_t> route_tail(vrp.size);

    // Doubly-linked list representation of routes
    std::vector<node_t> next_customer(vrp.size, DEPOT);
    std::vector<node_t> prev_customer(vrp.size, DEPOT);

    std::vector<node_t> temporary(vrp.size, -1);

    for (node_t i = 1; i < vrp.size; ++i)
    {
        route_demands[i] = vrp.node[i].demand;
        customer_route_map[i] = i;
        route_head[i] = i;
        route_tail[i] = i;
    }

    // --- 4. Merge Routes Greedily (with corrected O(1) merges) ---
    long long id = 0;
    for (const auto &saving : savings_list)
    {
        // std::cout << id++ << "\n";
        node_t i = saving.i;
        node_t j = saving.j;

        node_t route_id_i = customer_route_map[i];
        node_t route_id_j = customer_route_map[j];

        if (route_id_i != route_id_j)
        {
            if (route_demands[route_id_i] + route_demands[route_id_j] <= vrp.capacity)
            {
                node_t head_i = route_head[route_id_i];
                node_t tail_i = route_tail[route_id_i];
                node_t head_j = route_head[route_id_j];
                node_t tail_j = route_tail[route_id_j];

                bool merged = false;
                int reverse = -1;

                // Case 1: Tail of route i connects to Head of route j [...i] -> [j...]
                if (tail_i == i && head_j == j)
                {
                    next_customer[i] = j;
                    prev_customer[j] = i;
                    route_tail[route_id_i] = tail_j; // New tail is old tail of j
                    merged = true;
                }
                // Case 2: Tail of route j connects to Head of route i [...j] -> [i...]
                else if (tail_j == j && head_i == i)
                {
                    next_customer[j] = i;
                    prev_customer[i] = j;
                    route_head[route_id_i] = head_j; // New head is old head of j
                    merged = true;
                }
                // Case 3: Tail of i connects to Tail of j [...i] -> [...j](reversed)
                // else if (tail_i == i && tail_j == j)
                // {

                //     next_customer[j] = prev_customer[j];
                //     next_customer[i] = j;
                //     prev_customer[j] = i;
                //     route_tail[route_id_i] = head_j; // New tail is old head of j
                //     merged = true;
                //     reverse = 0;
                // }
                // // Case 4: Head of i connects to Head of j [i...](reversed) <- [j...]
                // else if (head_i == i && head_j == j)
                // {
                //     prev_customer[i] = next_customer[i];
                //     next_customer[j] = i;
                //     prev_customer[i] = j;
                //     route_head[route_id_i] = tail_j; // New head is old tail of j
                //     merged = true;
                //     reverse = 1;
                // }

                if (merged)
                {
                    route_demands[route_id_i] += route_demands[route_id_j];
                    customer_route_map[head_j] = route_id_i;
                    customer_route_map[tail_j] = route_id_i;
                    route_demands[route_id_j] = 0;

                    // for (node_t curr = 1; curr < vrp.getSize(); curr++)
                    // {
                    //     if (customer_route_map[curr] == route_id_j)
                    //     {
                    //         customer_route_map[curr] = route_id_i;
                    //     }
                    // }

                    route_head[route_id_j] = DEPOT;
                    route_tail[route_id_j] = DEPOT;
                }
            }
        }
    }

    // --- 5. Finalize and Reconstruct Routes ---
    std::vector<std::vector<node_t>> final_routes;
    std::vector<bool> visited_routes(vrp.size, false);

    for (node_t i = 1; i < vrp.size; ++i)
    {
        node_t route_id = customer_route_map[i];
        if (route_id != DEPOT && !visited_routes[route_id])
        {
            visited_routes[route_id] = true;
            std::vector<node_t> current_route;
            node_t current_node = route_head[route_id];
            while (current_node != DEPOT)
            {
                current_route.push_back(current_node);
                current_node = next_customer[current_node];
            }
            if (!current_route.empty())
            {
                final_routes.push_back(current_route);
            }
        }
    }
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
    if (argc > 2)
    {
        omp_set_num_threads(std::stoi(argv[2]));
    }

    VRP vrp;
    vrp.read(argv[1]);

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<node_t>> routes = parallel_savings_algorithm(vrp);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    weight_t total_cost = calCost(vrp, routes);
    routes = postProcessIt(vrp, routes, total_cost);
    total_cost = calCost(vrp, routes);
    bool is_valid = verify_sol(vrp, routes, vrp.getCapacity());
    for (auto &route : routes)
    {
        std::cout << "Route: 0 ";
        for (auto &node : route)
        {
            std::cout << node << " ";
        }
        std::cout << "0\n";
    }

    std::cout << "--- Parallel Clarke & Wright Savings Algorithm ---" << std::endl;
    std::cout << "Problem File: " << argv[1] << std::endl;
    std::cout << "Threads Used: " << omp_get_max_threads() << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Solution Cost: " << total_cost << std::endl;
    std::cout << "Total Time Taken:    " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID") << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}