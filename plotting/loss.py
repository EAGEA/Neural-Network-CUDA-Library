import matplotlib.pyplot as plt
import pandas as pd


def print_data(csv, ax, title):
    # Uses the first column for the x axes.
    csv.plot(y=csv.columns[0], marker='o', xticks=csv.iloc[:,0].astype(int), ax=ax)
    # Set the bottom value to 0 for the Y axes.
    ax.set_ylim(bottom=0)
    # Set the title.
    ax.set_title(title, fontsize=20)



def main():
    """
    Print the loss csv file obtained during the training
    of a neural network.
    The loss file (loss.csv) obtained from a neural network
    training (with option "print_loss" set), should be placed
    in the "data/" directory.
    """
    NB_ROWS = 1
    NB_COLS = 1
    XLABEL = "Number of (loss) records"
    YLABEL = "Loss value"

    fig, ax = plt.subplots(nrows=NB_ROWS, ncols=NB_COLS)
    fig.tight_layout()

    data = pd.read_csv("data/loss.csv", delimiter = ';')
    data.fillna(0.0, inplace=True)
    print(data)
    print_data(data, ax, "Evolution of the loss during the training")
    # Set the labels.
    plt.setp(ax, xlabel=XLABEL)
    plt.setp(ax, ylabel=YLABEL)
    # To have a graph that can be easily included in another document.
    plt.tight_layout(pad=0.05)
    # Full screen.
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    # Save into pdf.
    plt.savefig("loss", format="pdf", dpi=1200)
    # Show the graph.
    plt.show()



if __name__ == "__main__":
    main()
