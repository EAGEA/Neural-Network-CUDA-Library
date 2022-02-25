import matplotlib.pyplot as plt
import pandas as pd


def print_data(csv_1, csv_2, ax, title):
    # Uses the first column for the x axes.
    csv_1.plot(x=csv_1.columns[0], marker='o', xticks=[32,256,512], ax=ax)
    csv_2.plot(x=csv_2.columns[0], marker='x', xticks=[32,256,512], ax=ax)
    # Set the title.
    ax.set_title(title, fontsize=20)

    ax.set_yscale("log")



def main():
    """
    Print the CPU and GPU execution times obtained from
    the "lib/examples/op_time_functions.cpp" test (that records
    the execution time of every function).
    The file obtained from the test should be placed in the "data/"
    directory.
    """
    NB_OPERATIONS = 12 * 2
    NB_ROWS = 4
    NB_COLS = 6
    OPERATIONS = ["Linear",
                  "Linear derivative",
                  "Binary Step",
                  "Binary Step derivative",
                  "Sigmoid",
                  "Sigmoid derivative",
                  "Relu",
                  "Relu derivative",
                  "Tanh",
                  "Tanh derivative",
                  "Softmax",
                  "Softmax derivative",
                  "MSE",
                  "MSE derivative",
                  "MAE",
                  "MAE derivative",
                  "MBE",
                  "MBE derivative",
                  "Hinge Loss",
                  "Hinge Loss derivative",
                  "BCE",
                  "BCE derivative",
                  "CE",
                  "CE derivative"]
    XLABEL = "Size of a side of\nthe squared input matrices"
    YLABEL = "Execution time (ms)"

    fig, ax = plt.subplots(nrows=NB_ROWS, ncols=NB_COLS)
    plt.tight_layout()

    index_activation = 0
    index_loss = int(NB_OPERATIONS / 2)

    for i in range(0, NB_ROWS):
        for j in range(0, NB_COLS):
            index = 0 
            if j < NB_COLS / 2:
                index = index_activation
                index_activation += 1
            else: 
                index = index_loss
                index_loss += 1
            data_1 = pd.read_csv("data/" + str(index) + "_functions_gpu.csv", delimiter=';')
            data_2 = pd.read_csv("data/" + str(index) + "_functions_cpu.csv", delimiter=';')
            print_data(data_1, data_2, ax[i][j], OPERATIONS[index]) 

    # Set the labels.
    for i in range(0, NB_ROWS):
        plt.setp(ax[i, :], xlabel="")
    plt.setp(ax[-1, :], xlabel=XLABEL)
    plt.setp(ax[0:, 0], ylabel=YLABEL)
    plt.setp(ax[0:, int(NB_COLS/2)], ylabel=YLABEL)
    # Full screen.
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    # Save into pdf.
    plt.savefig("op_time_functions", format="pdf", dpi=1200)
    # Show the graph.
    plt.show()



if __name__ == "__main__":
    main()
