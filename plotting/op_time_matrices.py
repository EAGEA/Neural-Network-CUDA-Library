import matplotlib.pyplot as plt
import pandas as pd


def print_data(csv_1, csv_2, ax, title):
    # Uses the first column for the x axes.
    csv_1.plot(x=csv_1.columns[0], marker='o', xticks=[64,1024,2048], ax=ax)
    csv_2.plot(x=csv_2.columns[0], marker='x', xticks=[64,1024,2048], ax=ax)
    # Set the title.
    ax.set_title(title, fontsize=20)

    ax.set_yscale("log")



def main():
    """
    Print the CPU and GPU execution times obtained from
    the "lib/examples/op_time_matrices.cpp" test (that records
    the execution time of every matrix operation).
    The file obtained from this test should be placed in the "data/"
    directory. Then executing this script will produce a chart.
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
    XLABEL = "Size of a side of\nthe squared input matrices"
    YLABEL = "Execution time (ms)"

    fig, ax = plt.subplots(nrows=NB_ROWS, ncols=NB_COLS)
    plt.tight_layout(pad=3);

    for i in range(0, NB_ROWS):
        for j in range(0, NB_COLS):
            index = i * NB_COLS + j
            if index < NB_OPERATIONS:
                data_1 = pd.read_csv("data/" + str(index) + "_matrix_gpu.csv", delimiter=';')
                data_2 = pd.read_csv("data/" + str(index) + "_matrix_cpu.csv", delimiter=';')
                print_data(data_1, data_2, ax[i][j], OPERATIONS[index]) 
            else:
                ax[i][j].set_visible(False)

    # Set the labels.
    plt.setp(ax[:, :], xlabel=XLABEL)
    plt.setp(ax[:, :], ylabel=YLABEL)
    # Full screen.
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    # Save into pdf.
    plt.savefig("op_time_matrices", format="pdf", dpi=1200)
    # Show the graph.
    plt.show()



if __name__ == "__main__":
    main()
