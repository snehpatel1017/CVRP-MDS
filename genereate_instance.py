%%writefile generate_instance.py
import random
import math
import argparse

def generate_locations(num_customers, max_coord=100, distribution='R', num_clusters=5, cluster_std_dev=10):
    """
    Generates customer locations based on a specified distribution.

    Args:
        num_customers (int): The number of customers.
        max_coord (int): The maximum value for x and y coordinates.
        distribution (str): 'R' for Random, 'C' for Clustered, 'RC' for Random-Clustered.
        num_clusters (int): The number of cluster centers for 'C' and 'RC'.
        cluster_std_dev (int): The standard deviation for clusters.

    Returns:
        list: A list of (x, y) tuples for each customer.
    """
    locations = []
    if distribution in ['C', 'RC']:
        # 1. Create random cluster centers
        cluster_centers = [(random.randint(0, max_coord), random.randint(0, max_coord)) for _ in range(num_clusters)]

    if distribution == 'R': # Purely Random
        for _ in range(num_customers):
            locations.append((random.randint(0, max_coord), random.randint(0, max_coord)))

    elif distribution == 'C': # Purely Clustered
        for _ in range(num_customers):
            center_x, center_y = random.choice(cluster_centers)
            x = int(random.gauss(center_x, cluster_std_dev))
            y = int(random.gauss(center_y, cluster_std_dev))
            # Clamp coordinates to be within bounds
            locations.append((max(0, min(x, max_coord)), max(0, min(y, max_coord))))

    elif distribution == 'RC': # Random-Clustered
        clustered_customers = num_customers // 2
        random_customers = num_customers - clustered_customers
        # Generate clustered customers
        for _ in range(clustered_customers):
            center_x, center_y = random.choice(cluster_centers)
            x = int(random.gauss(center_x, cluster_std_dev))
            y = int(random.gauss(center_y, cluster_std_dev))
            locations.append((max(0, min(x, max_coord)), max(0, min(y, max_coord))))
        # Generate random customers
        for _ in range(random_customers):
            locations.append((random.randint(0, max_coord), random.randint(0, max_coord)))
        random.shuffle(locations)

    return locations

def generate_demands(num_customers, min_demand=1, max_demand=9):
    """Generates random integer demands for each customer."""
    return [random.randint(min_demand, max_demand) for _ in range(num_customers)]

def write_vrp_file(filename, name, capacity, locations, demands):
    """Writes the instance to a file in the standard TSPLIB (.vrp) format."""
    dimension = len(locations)
    with open(filename, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: CVRP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"CAPACITY: {capacity}\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(locations):
            f.write(f"{i + 1} {x} {y}\n")
        f.write("DEMAND_SECTION\n")
        f.write(f"1 0\n") # Depot has 0 demand
        for i, demand in enumerate(demands):
            f.write(f"{i + 2} {demand}\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")
    print(f"✅ Successfully generated instance: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate CVRP instances mimicking CVRPLIB benchmarks.")
    parser.add_argument("-n", type=int, default=100, help="Number of customers (e.g., 100).")
    parser.add_argument("-c", type=int, default=100, help="Vehicle capacity.")
    parser.add_argument("-t", type=str, choices=['R', 'C', 'RC'], default='C', help="Distribution type: R (Random), C (Clustered), RC (Random-Clustered).")
    parser.add_argument("--depot", type=str, choices=['center', 'random'], default='center', help="Depot position.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("-o", "--output", type=str, help="Output filename. If not provided, a name will be generated.")

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    customer_locations = generate_locations(args.n, distribution=args.t)
    demands = generate_demands(args.n)

    if args.depot == 'center':
        depot_location = (50, 50)
    else: # random
        depot_location = (random.randint(0, 100), random.randint(0, 100))

    all_locations = [depot_location] + customer_locations

    filename = args.output or f"{args.t}{args.n}_k_auto_C{args.c}.vrp"

    write_vrp_file(filename, filename.replace('.vrp', ''), args.c, all_locations, demands)

if __name__ == "__main__":
    main()