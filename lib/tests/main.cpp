//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include <cstdlib>
#include <ctime>


void main(int argc, char *argv[])
{
    // /!\ Init random generator.
    std::srand((unsigned int) std::time(nullptr));

    // Load and split the dataset.
    std::pair<dataset, dataset> split = dataset.load("path to").train_test_split();
    dataset train = split.first;
    dataset test = split.second;
    // Train and predict with a neural network.
    neural_network nn = neural_network(
            new layer(), ...
            );
    nn.fit(train_test.first);
    matrix predictions = nn.predict(train_test.second.get_features());

    return 0;
}