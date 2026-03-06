import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
# --------------------------------------------------
# Read CVRPLIB .vrp file
# --------------------------------------------------
def read_vrp(instance_path):
    coords = {}
    depot = None
    section = None

    with open(instance_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip empty lines
            if not line:
                continue

            if line == "NODE_COORD_SECTION":
                section = "coords"
                continue

            if line == "DEPOT_SECTION":
                section = "depot"
                continue

            if line == "EOF":
                break

            if section == "coords":
                parts = line.split()
                # Defensive check
                if len(parts) != 3:
                    continue
                node, x, y = parts
                coords[int(node) - 1] = (float(x), float(y))

            elif section == "depot":
                if line != "-1":
                    depot = int(line) - 1

    return depot, coords


# --------------------------------------------------
# Read solution routes file
# --------------------------------------------------
def read_routes(solution_path):
    routes = []
    cost = None

    with open(solution_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Cost"):
                cost = float(line.split(":")[1])

            elif line.startswith("Route"):
                nodes = list(map(int, line.split(":")[1].split()))
                routes.append(nodes)

    return cost, routes






# --------------------------------------------------
# Plot CVRP solution
# --------------------------------------------------


def plot_cvrp(depot, coords, routes, cost):
    plt.figure(figsize=(9, 9))

    for node, (x, y) in coords.items(): 
        if node == depot: 
            plt.scatter(x, y, c="red", s=300, marker="s", zorder=3) 
            plt.text(x, y, "Depot", fontsize=12, weight="bold") 
        else: 
            plt.scatter(x, y, c="black", s=60) 
            plt.text(x + 0.5, y + 0.5, str(node), fontsize=9)

    for route in routes: 
        color = (random.random(), random.random(), random.random()) 
        xs, ys = [], [] 
        
        for node in route: 
            x, y = coords[node] 
            xs.append(x) 
            ys.append(y) 
            plt.plot(xs, ys, linewidth=2.5, color=color)

    plt.title(f"CVRP Solution (Cost = {cost})") 
    plt.axis("equal") 
    plt.grid(True) 
    plt.show()



def plot_cvrp_large(depot, coords, routes, cost=None,
                    max_routes_drawn=500,
                    point_size=1,
                    point_alpha=0.15,
                    route_alpha=0.9):
    """
    Scalable CVRP visualizer.
    Designed for 10^5 – 10^6 nodes.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # -----------------------------
    # Plot ALL customers in ONE call
    # -----------------------------
    all_nodes = np.array(list(coords.values()))
    ax.scatter(
        all_nodes[:, 0],
        all_nodes[:, 1],
        s=point_size,
        c="black",
        alpha=point_alpha,
        linewidths=0,
        zorder=1
    )

    # -----------------------------
    # Plot depot
    # -----------------------------
    dx, dy = coords[depot]
    ax.scatter(
        dx, dy,
        c="red",
        s=120,
        marker="s",
        edgecolors="white",
        linewidths=1.5,
        zorder=3
    )

    # -----------------------------
    # Plot routes (as line segments)
    # -----------------------------
    cmap = cm.get_cmap("tab20", min(len(routes), max_routes_drawn))
    segments = []
    colors = []

    for r_idx, route in enumerate(routes[:max_routes_drawn]):
        route_coords = [coords[depot]] + [coords[n] for n in route] + [coords[depot]]

        for i in range(len(route_coords) - 1):
            segments.append([route_coords[i], route_coords[i + 1]])
            colors.append(cmap(r_idx))

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=0.7,
        alpha=route_alpha,
        zorder=2
    )
    ax.add_collection(lc)

    
    # Styling
    # -----------------------------
    title = "CVRP Solution"
    if cost is not None:
        title += f" (Cost = {cost:.2f})"

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.tight_layout()
    plt.show()



# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python visualize_cvrp.py <instance.vrp> <solution.txt>")
        sys.exit(1)
    type=0
    if len(sys.argv) == 4:
        type = int(sys.argv[3])
        

    instance_file = sys.argv[1]
    solution_file = sys.argv[2]
  

    

    depot, coords = read_vrp(instance_file)
    cost, routes = read_routes(solution_file)

   
    if type==1:
       plot_cvrp(depot, coords, routes, cost)
    else:
        # print("sdfd")
       plot_cvrp_large(depot,coords,routes,cost,max_routes_drawn=500)

