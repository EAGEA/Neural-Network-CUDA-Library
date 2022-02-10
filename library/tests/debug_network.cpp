//
// Created by Hugo on 10/02/2022.
//

#include "lib/data_structures/dataset/dataset.h"
#include "lib/functions/function.h"
#include "lib/models/neural_network/neural_network.h"
#include "lib/models/neural_network/layers/layer.h"

#include <cstdlib>
#include <ctime>
#include "chrono"

using namespace cudaNN;

using namespace std::chrono;


/**
 * Load a basic dataset. Define a simple neural network.
 * Train it with a part of the dataset, and predict with
 * another part. Print results.
 */
int main(int argc, char *argv[]) {
    // Init random generator.
    std::srand((unsigned int) std::time(nullptr));

    // Load and split the dataset.
    auto mult = dataset::load_mult();
    auto split = mult.train_test_split();
    auto train = split.first;
    //dataset::print(train);
    auto test = split.second;
    //dataset::print(test);
    // Define a basic neural network.
    neural_network nn = neural_network(
            {
                    new layer(dataset::MULT_NB_FEATURES, 8,
                              initializations::HE,
                              activation_functions::SIGMOID),
                    new layer(8, dataset::MULT_NB_LABELS,
                              initializations::XAVIER,
                              activation_functions::LINEAR)
            }
    );

    //Print the first layer at the beginning
    auto l = nn.get_layer(0);
    //Weights from this layer to the next
    l->print_weights();
    //Biases from this layer to the next
    l->print_biases();


    return EXIT_SUCCESS;
}