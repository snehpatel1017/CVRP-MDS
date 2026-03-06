import sys
import threading
import queue
import matplotlib.pyplot as plt

# --------------------------------------------------
# Read CVRPLIB .vrp file
# --------------------------------------------------
def read_vrp(instance_path):
    coords = {}
    depot = 1 # Default to 1 as per your specification
    section = None

    with open(instance_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

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
                if len(parts) >= 3:
                    node, x, y = parts[0], parts[1], parts[2]
                    # Store exact 1-based node IDs from file
                    coords[int(node)] = (float(x), float(y))

            elif section == "depot":
                if line != "-1":
                    depot = int(line)
                    section = None # Only read the first line of the depot section

    return depot, coords

# --------------------------------------------------
# Read the custom instructions .txt file
# --------------------------------------------------
def read_instructions(txt_path):
    instructions = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            if line == "commit":
                instructions.append("commit")
            else:
                parts = line.split()
                if len(parts) == 2:
                    instructions.append((int(parts[0]), int(parts[1])))
    return instructions

# --------------------------------------------------
# Interactive Plotting Function
# --------------------------------------------------
def interactive_plot(depot, coords, instructions, mode):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize the Star Graph: 
    # Every customer is initially in their own independent route [Depot -> Node -> Depot]
    routes = [[node] for node in coords.keys() if node != depot]

    def draw_state():
        ax.clear()
        
        # Plot all nodes
        x_vals = [pos[0] for pos in coords.values()]
        y_vals = [pos[1] for pos in coords.values()]
        ax.scatter(x_vals, y_vals, c="black", s=15, zorder=2)
        
        # Plot depot
        if depot in coords:
            dx, dy = coords[depot]
            ax.scatter(dx, dy, c="red", s=100, marker="s", zorder=3, label="Depot")
            
        # Draw current routes
        cmap = plt.get_cmap("tab20")
        for idx, route in enumerate(routes):
            if not route: continue
            
            # Wrap the route with the Depot at the start and end
            path = [depot] + route + [depot]
            
            x_pts = [coords[n][0] for n in path if n in coords]
            y_pts = [coords[n][1] for n in path if n in coords]
            
            color = cmap(idx % 20)
            ax.plot(x_pts, y_pts, color=color, linewidth=1.5, zorder=1)
            
        ax.set_title(f"Dynamic CVRP Graph (Mode {mode})")
        ax.set_aspect("equal", adjustable="box")
        plt.draw()
        plt.pause(0.01)

    # Initial Draw
    draw_state()

    def process_merge(u, v):
        # Apply the +1 offset as requested
        actual_u = u + 1
        actual_v = v + 1
        
        route_u_idx = -1
        route_v_idx = -1
        
        # Find which routes currently contain these nodes
        for i, r in enumerate(routes):
            if actual_u in r: route_u_idx = i
            if actual_v in r: route_v_idx = i
            
        # Merge the routes
        if route_u_idx != -1 and route_v_idx != -1 and route_u_idx != route_v_idx:
            # Connect the end of route U to the start of route V
            routes[route_u_idx].extend(routes[route_v_idx])
            # Delete the old separated route V
            del routes[route_v_idx]

    print("\n--- Interactive Graph Started ---")
    print("Initial state drawn (all customers connected to Depot).")
    print(f"Mode {mode} selected.")
    print("Press [ENTER] to advance, or type 'complete' to draw everything.")

    idx = 0
    total_inst = len(instructions)
    input_q = queue.Queue()

    def wait_for_input(prompt):
        """Asks for input in a background thread to prevent GUI freezing"""
        print(prompt, end="", flush=True)
        
        def get_input():
            input_q.put(sys.stdin.readline().strip().lower())
            
        threading.Thread(target=get_input, daemon=True).start()
        
        # Keep updating the GUI window while waiting for user keyboard input
        while input_q.empty():
            plt.pause(0.05) 
            
        return input_q.get()

    while idx < total_inst:
        user_input = wait_for_input(f"[{idx}/{total_inst}] Command: ")

        # Command to instantly finish the graph
        if user_input == 'complete':
            print("Completing the rest of the graph...")
            while idx < total_inst:
                inst = instructions[idx]
                if inst != "commit":
                    process_merge(inst[0], inst[1])
                idx += 1
            draw_state()
            break

        # Mode 1: One edge at a time
        if mode == 1:
            inst = instructions[idx]
            idx += 1
            if inst == "commit":
                print(" -> (Commit skipped in Mode 1)")
            else:
                process_merge(inst[0], inst[1])
                print(f" -> Merged nodes {inst[0]} and {inst[1]}")
            
            draw_state()

        # Mode 2: Advance until next 'commit'
        elif mode == 2:
            edges_drawn = 0
            while idx < total_inst:
                inst = instructions[idx]
                idx += 1
                
                if inst == "commit":
                    print(f" -> Reached commit. Merged {edges_drawn} pairs.")
                    break
                else:
                    process_merge(inst[0], inst[1])
                    edges_drawn += 1
                    
            draw_state()

    print("\nGraph complete. Close the matplotlib window to exit.")
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep window open until user closes it

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python visualize_dynamic.py <instance.vrp> <instructions.txt> <mode>")
        sys.exit(1)

    instance_file = sys.argv[1]
    instructions_file = sys.argv[2]
    mode = int(sys.argv[3])

    depot, coords = read_vrp(instance_file)
    instructions = read_instructions(instructions_file)

    interactive_plot(depot, coords, instructions, mode)