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
#include <cmath>   // For atan2

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
    vector<weight_t> dist;
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

    //~ 1  x1  y1
    //~ 2  x2  y2
    //~ 3  x3  y3
    //~ ...
    //~ n  xn  yn

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

        // assert(i==(id-1));
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

// START: SWEEP HEURISTIC IMPLEMENTATION
// This struct will hold a customer's ID and its angle relative to the depot.
struct CustomerAngle
{
    node_t id;
    double angle;

    // Overload the less-than operator to allow sorting by angle.
    bool operator<(const CustomerAngle &other) const
    {
        return angle < other.angle;
    }
};

/**
 * @brief Generates an initial customer tour using the Sweep Heuristic.
 *
 * The algorithm calculates the polar angle of each customer with respect to the
 * depot and sorts them based on this angle. This creates an ordered sequence
 * of customers, which forms a giant tour.
 *
 * @param vrp The VRP instance containing customer data.
 * @return A vector of node_t representing the ordered tour, starting with the depot.
 */
std::vector<node_t> SweepHeuristic(const VRP &vrp)
{
    std::vector<CustomerAngle> customer_angles;
    customer_angles.reserve(vrp.getSize() - 1);

    // Get depot coordinates.
    point_t depotX = vrp.node[DEPOT].x;
    point_t depotY = vrp.node[DEPOT].y;

    // Calculate the angle for each customer.
    for (size_t i = 0; i < vrp.getSize(); ++i)
    {
        if (i == DEPOT)
            continue; // Skip the depot itself.

        // Translate coordinates so the depot is at the origin.
        point_t dx = vrp.node[i].x - depotX;
        point_t dy = vrp.node[i].y - depotY;

        // Calculate angle using atan2 for quadrant correctness.
        double angle = atan2(dy, dx);
        customer_angles.push_back({(node_t)i, angle});
    }

    // Sort the customers based on their angle.
    std::sort(customer_angles.begin(), customer_angles.end());

    // Create the final single tour, starting with the depot.
    std::vector<node_t> singleRoute;
    singleRoute.reserve(vrp.getSize());
    singleRoute.push_back(DEPOT);
    for (const auto &ca : customer_angles)
    {
        singleRoute.push_back(ca.id);
    }

    return singleRoute;
}
// END: SWEEP HEURISTIC IMPLEMENTATION

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

std::vector<std::vector<node_t>>
Split_convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute)
{
    // Step 1: Create a tour of customers only, excluding the depot.
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

    // Step 2: Precomputation of prefix sums for demands and intra-tour distances.
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

    // Step 3: Initialize DP structures and the deque of optimal predecessors.
    std::vector<weight_t> V(n + 1, std::numeric_limits<weight_t>::max());
    std::vector<int> P(n + 1, -1);
    V[0] = 0; // Base case: cost to serve 0 customers is 0.

    std::deque<int> q;
    q.push_back(0);

    // Step 4: Main O(n) algorithm loop.
    for (int j = 1; j <= n; ++j)
    {
        // Capacity Pruning (Front): Remove predecessors from the front that are no longer capacity-feasible.
        while (!q.empty() && sum_demands[j] - sum_demands[q.front()] > vrp.getCapacity())
        {
            q.pop_front();
        }

        // Helper to calculate total cost via a predecessor `i` for a partition ending at `j`.
        // Cost = V[i] + Cost of route (Depot -> customer[i] -> ... -> customer[j-1] -> Depot)
        auto calculate_total_cost = [&](int i)
        {
            double route_dist;
            if (i == j - 1)
            { // Route with a single customer
                route_dist = vrp.get_dist(DEPOT, customer_tour[i]) + vrp.get_dist(customer_tour[i], DEPOT);
            }
            else
            {
                route_dist = vrp.get_dist(DEPOT, customer_tour[i])        // Depot to first customer
                             + (sum_dist[j] - sum_dist[i + 1])            // Intra-route travel
                             + vrp.get_dist(customer_tour[j - 1], DEPOT); // Last customer to Depot
            }
            return V[i] + route_dist;
        };

        // Dominance Pruning (Front): If the second candidate in the deque is better than the first for endpoint j, pop the first.
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

        // Set Potential and Predecessor: The best predecessor for j is now at the front.
        if (!q.empty())
        {
            P[j] = q.front();
            V[j] = calculate_total_cost(P[j]);
        }

        // Dominance Pruning (Back): Before adding j, remove candidates from the back that j dominates.
        // We compare their "quality score" g(i) = V[i] - sum_dist[i] + dist(Depot, customer[i-1]).
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

    // Step 5: Reconstruct the optimal routes by backtracking.
    std::vector<std::vector<node_t>> final_routes;
    int current_idx = n;
    while (current_idx > 0)
    {
        int pred_idx = P[current_idx];
        std::vector<node_t> new_route;
        // The route contains customers from index pred_idx to current_idx-1
        for (int k = pred_idx; k < current_idx; ++k)
        {
            new_route.push_back(customer_tour[k]);
        }
        final_routes.push_back(new_route);
        current_idx = pred_idx;
    }

    // The routes were built backward, so reverse the final list.
    std::reverse(final_routes.begin(), final_routes.end());
    return final_routes;
}
// Cost of a CVRP Solution!.
weight_t calRouteValue(const VRP &vrp, const std::vector<node_t> &aRoute, node_t depot = 1)
{ // return cost of "a" route
    weight_t routeVal = 0;
    node_t prevPoint = 0; // First point in a route is depot

    for (auto aPoint : aRoute)
    {
        routeVal += vrp.get_dist(prevPoint, aPoint);
        prevPoint = aPoint;
    }
    routeVal += vrp.get_dist(prevPoint, 0); // Last point in a route is depot

    return routeVal;
}

// Print in DIMACS output format http://dimacs.rutgers.edu/programs/challenge/vrp/cvrp/
// Depot is 0
// Route #1: 1 2 3
// Route #2: 4 5
// ...
// Route #k: n-1 n
//
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
    // 'cities' contains the original solution. It is updated during the course of the 2opt-scheme to contain the 2opt soln.
    // 'tour' is an auxillary array.

    // repeat until no improvement is made
    unsigned improve = 0;

    while (improve < 2)
    {
        double best_distance = 0.0;

        best_distance += vrp.get_dist(DEPOT, cities[0]); // computing distance of the first point in the route with the depot.

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
                    // Improvement found so reset
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
        // postprocessing final_routes[i]
        unsigned sz = final_routes[i].size();
        //~ unsigned* cities = (unsigned*) malloc(sizeof(unsigned) * (sz));
        //~ unsigned* tour = (unsigned*) malloc(sizeof(unsigned) * (sz));  // this is an auxillary array

        std::vector<node_t> cities(sz);
        std::vector<node_t> tour(sz);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = final_routes[i][j];

        vector<node_t> curr_route;

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

weight_t get_total_cost_of_routes(const VRP &vrp, vector<vector<node_t>> &final_routes)
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

        vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
        vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];

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

// MAIN function.
// Some debugging/print functions are commented out.
int main(int argc, char *argv[])
{
    VRP vrp;
    if (argc < 2)
    {
        std::cout << "parMDS version 1.1 (Sweep Heuristic)" << '\n';
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

    // START TIMER
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // MODIFICATION: Replace MST + DFS with Sweep Heuristic
    // =========================================================================

    // OLD MST-based approach is removed.
    // auto mstG = PrimsAlgo(vrp);
    // std::vector<bool> visited(mstG.size(), false);
    // visited[0] = true;
    // std::vector<int> singleRoute;
    // ShortCircutTour(mstG, visited, 0, singleRoute);

    // NEW: Generate the initial tour using the Sweep Heuristic
    std::vector<node_t> singleRoute = SweepHeuristic(vrp);

    // Split the single giant tour into a set of feasible routes using the optimal splitting algorithm.
    auto initialRoutes = Split_convertToVrpRoutes(vrp, singleRoute);

    // Calculate the cost of this initial solution.
    auto costRoutePair = calCost(vrp, initialRoutes);
    weight_t initialCost = costRoutePair.first;
    std::vector<std::vector<node_t>> bestRoutes = costRoutePair.second;

    // =========================================================================
    // END OF MODIFICATION
    // =========================================================================

    // The large randomization loop is removed because the Sweep Heuristic is deterministic.
    // We get one good initial solution and proceed directly to post-processing.

    std::chrono::high_resolution_clock::time_point end_initial = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_initial = std::chrono::duration_cast<std::chrono::nanoseconds>(end_initial - start).count();
    double timeForInitialSolution = (double)(elapsed_initial * 1.E-9);

    std::chrono::high_resolution_clock::time_point start_postprocess = std::chrono::high_resolution_clock::now();

    weight_t finalCost = initialCost;
    auto postprocessedRoutes = postProcessIt(vrp, bestRoutes, finalCost);

    // END TIMER ALL
    std::chrono::high_resolution_clock::time_point end_total = std::chrono::high_resolution_clock::now();
    elapsed_initial = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_postprocess).count();
    double timeForPostProcessing = (double)(elapsed_initial * 1.E-9);

    elapsed_initial = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start).count();
    double total_time = (double)(elapsed_initial * 1.E-9);

    /// VALIDATION
    bool verified = false;
    verified = verify_sol(vrp, postprocessedRoutes, vrp.getCapacity());

    // Print final results
    std::cout << argv[1] << " ";
    std::cout << "Initial Cost (Sweep) = " << initialCost << ',';
    std::cout << "Final Cost (Post-Processed) = " << finalCost;
    std::cout << " | Time(s): ";
    std::cout << "Initial Solution = " << timeForInitialSolution << ',';
    std::cout << "Post-Processing = " << timeForPostProcessing << ',';
    std::cout << "Total = " << total_time;

    if (verified)
        std::cout << " VALID" << std::endl;
    else
        std::cout << " INVALID" << std::endl;

    // PRINT Final Answer Routes if needed
    // printOutput(vrp, postprocessedRoutes);

    return 0;
}