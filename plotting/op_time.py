import matplotlib.pyplot as plt
import pandas as pd


def print_data(csv_1, csv_2, ax, title):
    """
    Use the data from the csv files to print two graphs; one
    for the CPU execution time, and another for the GPU.
    """
    # Uses the first column for the x axes.
    csv_1.plot(x = csv_1.columns[0], marker='o', xticks=csv_1.iloc[:,0], ax=ax)
    csv_2.plot(x = csv_2.columns[0], marker='x', xticks=csv_2.iloc[:,0], ax=ax)
    # Set the bottom value to 0 for the Y axes.
    ax.set_ylim(bottom=0)
    # Set the title.
    ax.set_title(title, fontsize=20)



def main():
    """
    Print the CPU and GPU execution times obtained from
    the "lib/examples/op_time.cpp" test (that records
    the execution time of every matrix operation).
    The file obtained from the test should be placed in the "data/"
    directory.
    """
    NB_OPERATIONS = 7
    NB_ROWS = 2
    NB_COLS = 4
    OPERATIONS = ["Addition\nof 2 matrices", 
                  "Subtraction\nof 2 matrices",
                  "Multiplication\nof 2 matrices", 
                  "Multiplication of a\nmatrix and a float",
                  "Hadamard product\nbetween 2 matrices", 
                  "Sum of the values\n of a matrix",
                  "Transpose\n of a matrix"]
    XLABEL = "Elements in the matrices (2^N)"
    YLABEL = "Execution time (ms)"

    fig, ax = plt.subplots(nrows=NB_ROWS, ncols=NB_COLS)
    fig.tight_layout();

    for i in range(0, NB_ROWS):
        for j in range(0, NB_COLS):
            index = i * NB_COLS + j
            if index < NB_OPERATIONS:
                data_1 = pd.read_csv("data/" + str(index) + "_gpu.csv", delimiter = ';')
                data_2 = pd.read_csv("data/" + str(index) + "_cpu.csv", delimiter = ';')
                print_data(data_1, data_2, ax[i][j], OPERATIONS[index]) 
            else:
                ax[i][j].set_visible(False)

    # Set the labels.
    plt.setp(ax[0, :], xlabel=XLABEL)
    plt.setp(ax[1, :], xlabel=XLABEL)
    plt.setp(ax[:, 0], ylabel=YLABEL)
    # To have a graph that can be easily included in another document.
    plt.tight_layout(pad=0.05)
    # Full screen.
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    # Save into pdf.
    plt.savefig("op_time", format="pdf", dpi=1200)
    # Show the graph.
    plt.show()



if __name__ == "__main__":
    main()
