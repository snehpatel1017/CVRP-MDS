#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <random>
#include <list>
#include <cstring>

// --- Data Structures ---
using node_t = int;
using weight_t = double;
using demand_t = double;

const node_t DEPOT = 0;

struct Point
{
    double x, y, demand;
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
    weight_t get_dist(const Point &p1, const Point &p2) const;
    size_t getSize() const
    {
        return size;
    }
};

// --- Helper: Distance Calculation ---
weight_t VRP::get_dist(node_t i, node_t j) const
{
    double dx = node[i].x - node[j].x;
    double dy = node[i].y - node[j].y;
    return std::sqrt(dx * dx + dy * dy);
}

weight_t VRP::get_dist(const Point &p1, const Point &p2) const
{
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// --- IO: Reading Input (Same as your code) ---
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

bool verify_sol(const VRP &vrp, std::vector<std::vector<node_t>> final_routes, unsigned capacity)
{
    /* verifies if the solution is valid or not */
    /**
     * 1. All vertices appear in the solution exactly once.
     * 2. For every route, the capacity constraint is respected.
     **/

    unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
    std::memset(hist, 0, sizeof(unsigned) * vrp.getSize());

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

// ==========================================================
// CLUSTERING (Rebalancing & Smoothing)
// ==========================================================

struct Cluster
{
    Point centroid;
    std::vector<node_t> members;
    double current_load;
};

void update_centroid(const VRP &vrp, Cluster &c)
{
    if (c.members.empty())
        return;
    double sx = 0, sy = 0;
    for (node_t member : c.members)
    {
        sx += vrp.node[member].x;
        sy += vrp.node[member].y;
    }
    c.centroid.x = sx / c.members.size();
    c.centroid.y = sy / c.members.size();
}

std::vector<Cluster> initialize_clusters(const VRP &vrp, int K)
{
    std::vector<Cluster> clusters(K);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, vrp.size - 1);

    int first_idx = dist(rng);
    clusters[0].centroid = vrp.node[first_idx];

    for (int k = 1; k < K; ++k)
    {
        double max_dist = -1.0;
        int best_candidate = -1;
        for (int i = 0; i < 100; ++i)
        {
            int candidate = dist(rng);
            double min_dist_to_centers = DBL_MAX;
            for (int j = 0; j < k; ++j)
            {
                double d = vrp.get_dist(vrp.node[candidate], clusters[j].centroid);
                if (d < min_dist_to_centers)
                    min_dist_to_centers = d;
            }
            if (min_dist_to_centers > max_dist)
            {
                max_dist = min_dist_to_centers;
                best_candidate = candidate;
            }
        }
        if (best_candidate == -1)
            best_candidate = dist(rng);
        clusters[k].centroid = vrp.node[best_candidate];
    }
    return clusters;
}

// Rebalancing logic using MAX_CLUSTER_LOAD instead of just CAPACITY
void rebalance_clusters(const VRP &vrp, std::vector<Cluster> &clusters, double max_cluster_load)
{
    bool changed = true;
    int iter = 0;
    int max_rebalance_iters = 30;

    while (changed && iter < max_rebalance_iters)
    {
        changed = false;
        iter++;
        for (auto &c : clusters)
            update_centroid(vrp, c);

        for (size_t source_idx = 0; source_idx < clusters.size(); ++source_idx)
        {
            for (int i = clusters[source_idx].members.size() - 1; i >= 0; --i)
            {
                node_t node_id = clusters[source_idx].members[i];
                double current_dist = vrp.get_dist(vrp.node[node_id], clusters[source_idx].centroid);

                int best_target_idx = -1;
                double best_improvement = 0.0;

                for (size_t target_idx = 0; target_idx < clusters.size(); ++target_idx)
                {
                    if (source_idx == target_idx)
                        continue;
                    double target_dist = vrp.get_dist(vrp.node[node_id], clusters[target_idx].centroid);

                    if (target_dist < current_dist)
                    {
                        // CHECK AGAINST MAX_CLUSTER_LOAD (k * C * util)
                        if (clusters[target_idx].current_load + vrp.node[node_id].demand <= max_cluster_load)
                        {
                            double improvement = current_dist - target_dist;
                            if (improvement > 0.001 && improvement > best_improvement)
                            {
                                best_improvement = improvement;
                                best_target_idx = target_idx;
                            }
                        }
                    }
                }

                if (best_target_idx != -1)
                {
                    clusters[source_idx].current_load -= vrp.node[node_id].demand;
                    clusters[source_idx].members.erase(clusters[source_idx].members.begin() + i);
                    clusters[best_target_idx].current_load += vrp.node[node_id].demand;
                    clusters[best_target_idx].members.push_back(node_id);
                    changed = true;
                }
            }
        }
    }
}

// Clustering: Input k (routes per cluster) and util
std::vector<std::vector<node_t>> cluster_nodes(const VRP &vrp, int k_routes_per_cluster, double target_utilization)
{
    double total_demand = 0;
    for (size_t i = 1; i < vrp.size; ++i)
        total_demand += vrp.node[i].demand;

    // The maximum load allowed in one cluster
    double max_cluster_load = vrp.capacity * k_routes_per_cluster * target_utilization;

    // Calculate how many clusters we need
    // Total Clusters = Total Demand / (Capacity per Cluster)
    int K = std::ceil(total_demand / max_cluster_load);
    // Safety check: ensure K is at least 1
    if (K < 1)
        K = 1;

    std::cout << "Clustering into " << K << " sectors (Max Load per cluster: " << max_cluster_load << ")..." << std::endl;
    std::vector<Cluster> clusters = initialize_clusters(vrp, K);

    int MAX_ITER = 1000;
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        for (auto &c : clusters)
        {
            c.members.clear();
            c.current_load = 0;
        }

        std::vector<int> nodes_to_assign;
        for (size_t i = 1; i < vrp.size; ++i)
            nodes_to_assign.push_back(i);
        std::shuffle(nodes_to_assign.begin(), nodes_to_assign.end(), std::mt19937(iter));

        for (int node_idx : nodes_to_assign)
        {
            double best_dist = DBL_MAX;
            int best_cluster = -1;

            // Find closest VALID cluster
            for (int k = 0; k < K; ++k)
            {
                double d = vrp.get_dist(vrp.node[node_idx], clusters[k].centroid);
                if (d < best_dist)
                {
                    if (clusters[k].current_load + vrp.node[node_idx].demand <= max_cluster_load)
                    {
                        best_dist = d;
                        best_cluster = k;
                    }
                }
            }

            // Fallback if all valid are full
            if (best_cluster == -1)
            {
                best_dist = DBL_MAX;
                for (int k = 0; k < K; ++k)
                {
                    double d = vrp.get_dist(vrp.node[node_idx], clusters[k].centroid);
                    if (d < best_dist)
                    {
                        best_dist = d;
                        best_cluster = k;
                    }
                }
            }

            clusters[best_cluster].members.push_back(node_idx);
            clusters[best_cluster].current_load += vrp.node[node_idx].demand;
        }

        double shift = 0.0;
        for (auto &c : clusters)
        {
            if (c.members.empty())
                continue;
            Point old_c = c.centroid;
            update_centroid(vrp, c);
            shift += vrp.get_dist(old_c, c.centroid);
        }
        if (shift < 1.0)
            break;
    }

    rebalance_clusters(vrp, clusters, max_cluster_load);

    // Output only members
    std::vector<std::vector<node_t>> result;
    for (const auto &c : clusters)
    {
        if (!c.members.empty())
            result.push_back(c.members);
    }
    return result;
}

// ==========================================================
// ROUTING: Split Cluster into k (or roughly k) Routes
// ==========================================================

// Nearest Neighbor Logic to split one large cluster into multiple valid routes
std::vector<std::vector<node_t>> split_cluster_into_routes_nn(const VRP &vrp, std::vector<node_t> cluster_nodes)
{
    std::vector<std::vector<node_t>> cluster_routes;
    std::vector<bool> visited(cluster_nodes.size(), false);
    int nodes_processed = 0;

    // While there are unvisited nodes in this cluster
    while (nodes_processed < cluster_nodes.size())
    {
        std::vector<node_t> current_route;
        double current_load = 0;
        node_t current_loc = DEPOT;

        while (true)
        {
            int best_idx = -1;
            double best_dist = DBL_MAX;

            // Find nearest unvisited neighbor in this cluster that fits capacity
            for (size_t i = 0; i < cluster_nodes.size(); ++i)
            {
                if (!visited[i])
                {
                    node_t candidate = cluster_nodes[i];
                    if (current_load + vrp.node[candidate].demand <= vrp.capacity)
                    {
                        double d = vrp.get_dist(current_loc, candidate);
                        if (d < best_dist)
                        {
                            best_dist = d;
                            best_idx = i;
                        }
                    }
                }
            }

            // If we found a valid node
            if (best_idx != -1)
            {
                visited[best_idx] = true;
                node_t node_id = cluster_nodes[best_idx];
                current_route.push_back(node_id);
                current_load += vrp.node[node_id].demand;
                current_loc = node_id;
                nodes_processed++;
            }
            else
            {
                // No more nodes fit in this vehicle, OR no nodes left at all
                break;
            }
        }
        if (!current_route.empty())
        {
            cluster_routes.push_back(current_route);
        }
        else
        {

            break;
        }
    }
    return cluster_routes;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename.vrp> <k_routes_per_cluster> <target_utilization>" << std::endl;
        std::cerr << "Example: " << argv[0] << " input.vrp 2 0.95" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int k_routes = std::atoi(argv[2]);
    double target_util = std::atof(argv[3]);

    if (k_routes < 1)
        k_routes = 1;
    if (target_util <= 0.0 || target_util > 1.0)
        target_util = 1.0;

    VRP vrp;
    vrp.read(filename);

    auto start_time = std::chrono::high_resolution_clock::now();

    // STEP 1: Cluster (Decompose into larger sectors)
    // Each cluster now holds approx k_routes worth of demand
    std::vector<std::vector<node_t>> raw_clusters = cluster_nodes(vrp, k_routes, target_util);

    // STEP 2: Atomic Routing (Split Clusters -> Routes)
    std::vector<std::vector<node_t>> all_routes;

    for (size_t i = 0; i < raw_clusters.size(); ++i)
    {
        // Split the large cluster into actual vehicle routes using NN
        std::vector<std::vector<node_t>> sub_routes = split_cluster_into_routes_nn(vrp, raw_clusters[i]);

        {
            // Flatten logic: Add sub-routes to the main list
            for (const auto &r : sub_routes)
            {
                all_routes.push_back(r);
            }
        }
    }

    // STEP 3: Post-Processing Optimization (TSP 2-Opt on individual routes)
    weight_t minCost;
    auto final_routes = postProcessIt(vrp, all_routes, minCost);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    auto total_cost = calCost(vrp, final_routes);
    bool is_valid = verify_sol(vrp, final_routes, vrp.capacity);

    std::cout << "--- Cluster-First Route-Second (Multi-Route Clusters) ---" << std::endl;
    std::cout << "Problem File:       " << argv[1] << std::endl;
    std::cout << "K (Routes/Cluster): " << k_routes << std::endl;
    std::cout << "Target Util:        " << target_util << std::endl;
    std::cout << "Generated Routes:   " << final_routes.size() << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Solution Cost: " << total_cost.first << std::endl;
    std::cout << "Total Time Taken:    " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Solution Validity:   " << (is_valid ? "VALID" : "INVALID") << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}
