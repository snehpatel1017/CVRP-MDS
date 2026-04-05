import matplotlib.pyplot as plt

def plot_costs_1(filename):
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

def plot_costs_2(filename):
    costs1=[]
    costs2 = []
    customers = []
    # Read the first file and convert each line to a float
    try:
        with open(filename, 'r') as f1:
            itr=0
            for line in f1:
                if itr==0:
                    itr+=1
                    continue
                itr+=1
                clean_line = line.strip()
                segmenst = clean_line.split(',')
                print(segmenst)
                costs1.append(float(segmenst[2]))
                costs2.append(float(segmenst[5]))
                customers.append(int(segmenst[6]))
                
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    
    zipped_list = zip(costs1, costs2, customers)
    sorted_zipped_list = sorted(zipped_list)
    costs1, costs2, customers = zip(*sorted_zipped_list)

    

    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(customers, costs1, marker='o',  color='b', label='C&W Cost')
    plt.scatter(customers, costs2, marker='o',  color='r', label='Rajesh Code Cost')
    
    # Adding labels and title
    plt.xlabel('Customers (x)')
    plt.ylabel('Cost (y)')
    plt.title('Cost comparision between C&W and Rajesh Code')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Replace 'costs.txt' with your actual filename
    filename = "./Og_vs_Rajesh.csv"
    # plot_costs('costs.txt')
    plot_costs_2(filename)