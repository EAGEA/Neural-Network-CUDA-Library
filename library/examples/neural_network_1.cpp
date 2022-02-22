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
                              activation_functions::RELU),
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
    neural_network::print(nn);
    // Train the neural network and record the time.
    float time;
    util::CPU_start_record(&time);
    nn.fit(train, loss_functions::BINARY_CROSS_ENTROPY_LOSS,
           2,16, 0.01);
    util::CPU_end_record(&time);
    // Predict using the test dataset.
    auto predictions = nn.predict(test);
    // Print ground truths and the predictions, compute accuracy on this test.
    auto correct = 0;
    for (size_t i = 0; i < predictions.size(); i ++)
    {
        std::cout << "-------------------------------------------" << std::endl;
        matrix::print(test.get(i).get_labels());
        matrix::print(predictions[i]);
    }
    // Logs.
    std::cout << std::endl
              << "TRAIN TIME: "
              << time / 1000.f
              << " seconds"
              << std::endl;
    std::cout << "ACCURACY: "
              << (int) (100.f * (float) correct / (float) predictions.size())
              << "%"
              << std::endl;

    return EXIT_SUCCESS;
}