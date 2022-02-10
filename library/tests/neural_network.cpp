//
// Created by Emilien Aufauvre on 29/10/2021.
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
                    new layer(dataset::MULT_NB_FEATURES, 2048,
                              initializations::HE,
                              activation_functions::SIGMOID),
                    new layer(2048, 1024,
                              initializations::XAVIER,
                              activation_functions::TANH),
                    new layer(1024, 512,
                              initializations::HE,
                              activation_functions::RELU),
                    new layer(512, 256,
                              initializations::HE,
                              activation_functions::RELU),
                    new layer(256, dataset::MULT_NB_LABELS,
                              initializations::XAVIER,
                              activation_functions::LINEAR)
            }
    );
    // Train the neural network.
    auto start = high_resolution_clock::now();
    nn.fit(train, loss_functions::BINARY_CROSS_ENTROPY_LOSS,
           2,16, 0.01);
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "TOTAL TIME : " << diff.count() << " seconds.\n";
    // Predict using the test dataset.
    auto predictions = nn.predict(test);
    // Print ground truths and the predictions.
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i ++)
    {
        /*std::cout << "-------------------------------------------" << std::endl;
        matrix::print(test.get(i).get_labels());
        matrix::print(predictions[i]);*/
        if(test.get(i).get_labels() == predictions[i])
        {
            correct+=1;
        }
    }
    std::cout << "Accuracy : " << correct/predictions.size() << "%\n";

    return EXIT_SUCCESS;
}