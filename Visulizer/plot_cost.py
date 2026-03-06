import matplotlib.pyplot as plt

def plot_costs(filename):
    costs = []
    
    # Read the file and convert each line to a float
    try:
        with open(filename, 'r') as file:
            for line in file:
                # strip() removes whitespace; float() handles '1.7e11' automatically
                clean_line = line.strip()
                if clean_line:
                    costs.append(float(clean_line))
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return

    # Generate x-axis values starting from 1
    iterations = range(1, len(costs) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, costs, marker='o', linestyle='-', color='b', markersize=4)
    
    # Adding labels and title
    plt.xlabel('Iteration (x)')
    plt.ylabel('Cost (y)')
    plt.title('Cost Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Replace 'costs.txt' with your actual filename
    plot_costs('costs.txt')