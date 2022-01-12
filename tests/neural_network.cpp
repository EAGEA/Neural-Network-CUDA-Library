//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "lib/data_structures/dataset/dataset.h"
#include "lib/models/neural_network/functions/function.h"
#include "lib/models/neural_network/neural_network.h"
#include "lib/models/neural_network/layers/layer.h"

#include <cstdlib>
#include <ctime>


using namespace cudaNN;


/**
 * Load a basic dataset. Define a simple neural network.
 * Train it with a part of the dataset, and predict with
 * another part. Print results.
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
                    new layer(dataset::MULT_NB_FEATURES, 16,
                              initializations::HE,
                              activation_functions::RELU),
                    new layer(16, 8,
                              initializations::XAVIER,
                              activation_functions::TANH),
                    new layer(8, 16,
                              initializations::HE,
                              activation_functions::RELU),
                    new layer(16, dataset::MULT_NB_LABELS,
                              initializations::XAVIER,
                              activation_functions::LINEAR)
            }
    );
    // Train the neural network.
    nn.fit(train, loss_functions::MEAN_SQUARED_ERROR, 2);
    // Predict using the test dataset.
    auto predictions = nn.predict(test);
    // Print ground truths and the predictions.
    for (size_t i = 0; i < predictions.size(); i ++)
    {
        std::cout << "-------------------------------------------" << std::endl;
        matrix::print(test.get(i).get_labels());
        matrix::print(predictions[i]);
    }

    return EXIT_SUCCESS;
}