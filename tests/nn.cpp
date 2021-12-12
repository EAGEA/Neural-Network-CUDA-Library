//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "lib/datastructs/dataset/dataset.h"
#include "lib/models/neural_network/neural_network.h"
#include "lib/models/neural_network/layers/linear_layer.h"
#include "lib/parameters/activation_functions/activation_functions.h"

#include <cstdlib>
#include <ctime>
#include <utility>


using namespace cudaNN;


/**
 * TODO DEFINE TEST
 */
int main(int argc, char *argv[])
{
    // /!\ Init random generator.
    std::srand((unsigned int) std::time(nullptr));

    // Load and split the dataset.
    std::pair<dataset, dataset> split = dataset::load_mult().train_test_split();
    dataset train = split.first;
    dataset test = split.second;
    // Train and predict with a neural network.
    neural_network nn = neural_network(
            {
                new linear_layer(dataset::MULT_NB_FEATURES, 16, 
                        activation_functions::linear),
                new linear_layer(16, dataset::MULT_NB_LABELS, 
                        activation_functions::sigmoid)
            }
        );
    nn.fit(train, &loss_functions::mean_square_error);
    matrix predictions = nn.predict(test.get_features());

    return EXIT_SUCCESS;
}
