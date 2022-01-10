//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "lib/data_structures/dataset/dataset.h"
#include "lib/models/neural_network/neural_network.h"
#include "lib/models/neural_network/layers/layer.h"
#include "lib/parameters/activation_functions/activation_function_.h"

#include <cstdlib>
#include <ctime>
#include <utility>


using namespace cudaNN;


/**
 * Load a basic dataset. Define a simple neural network.
 * Train it with a part of the dataset, and predict with
 * another part.
 */
int main(int argc, char *argv[])
{
    // Init random generator.
    std::srand((unsigned int) std::time(nullptr));

    // Load and split the dataset.
    auto mult = dataset::load_mult();
    auto split = mult.train_test_split();
    auto train = split.first;
    auto test = split.second;
    // Define a basic neural network.
    neural_network nn = neural_network(
            {
                    new layer(dataset::MULT_NB_FEATURES, 20,
                              activation_functions::linear),
                    new layer(20, dataset::MULT_NB_LABELS,
                              activation_functions::relu)
            }
    );
    // Train the neural network.
    nn.fit(train, loss_functions::mean_squared_error);
    // Predict using the test dataset.
    auto predictions = nn.predict(test);
    // Show ground truth and the predictions.
    for (size_t i = 0; i < predictions.size(); i ++)
    {
        matrix::print(test.get(i).get_labels());
        matrix::print(predictions[i]);
    }

    return EXIT_SUCCESS;
}