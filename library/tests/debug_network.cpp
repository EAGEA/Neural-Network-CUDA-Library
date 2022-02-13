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
    auto mult = dataset::load_smallimg();
    auto split = mult.train_test_split();
    auto train = split.first;
    //dataset::print(train);
    auto test = split.second;
    //dataset::print(test);
    // Define a basic neural network.
    neural_network nn = neural_network(
            {
                    new layer(dataset::SMALLIMG_NB_FEATURES, 8,
                              initializations::HE,
                              activation_functions::SIGMOID),
                    new layer(8, dataset::SMALLIMG_NB_LABELS,
                              initializations::XAVIER,
                              activation_functions::SOFTMAX)
            }
    );

    //Print the first layer at the beginning
    auto l = nn.get_layer(1);
    //Weights from this layer to the next
    l->print_weights();
    //Biases from this layer to the next
    //l->print_biases();

    // Train the neural network.
    auto start = high_resolution_clock::now();
    nn.fit(train, loss_functions::CROSS_ENTROPY_LOSS,
           1,1, 0.01);
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "TOTAL TIME : " << diff.count() << " seconds.\n";

    // Predict using the test dataset.
    auto predictions = nn.predict(test);
    // Print ground truths and the predictions.
    float correct = .0f;
    for (size_t i = 0; i < predictions.size(); i ++)
    {
        std::cout << "-------------------------------------------" << std::endl;
        matrix::print(test.get(i).get_labels());
        matrix::print(predictions[i]);
        float max = .0f;
        int id = 0;
        for(int j = 0; j < predictions[i].get_length(); j++)
        {
            if(predictions[i][j] >= max)
            {
                max = predictions[i][j];
                id = j;
            }
        }
        if(test.get(i).get_labels()[id] == 1)
        {
            correct+=.1f;
        }
    }
    std::cout << "Accuracy : " << (correct/predictions.size()) * 1000 << "%\n";


    return EXIT_SUCCESS;
}