#include "elsa.h"

#include <iostream>

using namespace elsa;

std::vector<DataContainer<real_t>> loadData(const std::string& fileName)
{
    std::vector<DataContainer<real_t>> trainingImages;

    std::ifstream myfile;
    std::string line;
    myfile.open(fileName);
    if (myfile.is_open()) {
        printf("opened %s\n", fileName.c_str());
    }
    std::getline(myfile, line);
    while (!myfile.eof()) {
        DataContainer<real_t> vectorizedImage(VolumeDescriptor{{28 * 28}});
        int pixelDensity = 0;
        int pixelIndex = 0;
        for (char c : line) {
            if (c == ',') {
                //                printf("num is %d\n", pixelDensity);
                vectorizedImage[pixelIndex++] = static_cast<real_t>(pixelDensity);
                pixelDensity = 0;
                continue;
            }
            pixelDensity = pixelDensity * 10 + (int) c - '0';
        }
        trainingImages.push_back(vectorizedImage);
        std::getline(myfile, line);
    }
    myfile.close();

    return trainingImages;
}

std::vector<DataContainer<real_t>> loadLabels(const std::string& fileName)
{
    std::vector<DataContainer<real_t>> trainingLabels;
    std::ifstream myfile;
    std::string line;
    myfile.open(fileName);
    if (myfile.is_open()) {
        printf("opened %s\n", fileName.c_str());
    }
    std::getline(myfile, line);
    while (!myfile.eof()) {
        char c = line[0];
        DataContainer<real_t> imageLabel(VolumeDescriptor{{1}});
        //        printf("%d\n", (int) c - '0');
        imageLabel = static_cast<real_t>((int) c - '0');
        trainingLabels.push_back(imageLabel);
        std::getline(myfile, line);
    }
    myfile.close();

    return trainingLabels;
}

void mnist_example()
{
    //    VolumeDescriptor inputDescriptor{{28 * 28}};
    //
    //    auto input = ml::Input(inputDescriptor, 1);
    //
    //    auto flatten = ml::Flatten();
    //    flatten.setInput(&input);
    //
    //    // A dense layer with 128 neurons and Relu activation
    //    auto dense = ml::Dense(128, ml::Activation::Relu);
    //    dense.setInput(&flatten);
    //
    //    // A dense layer with 10 neurons and Relu activation
    //    auto dense2 = ml::Dense(10, ml::Activation::Relu);
    //    dense2.setInput(&dense);
    //
    //    auto softmax = ml::Softmax();
    //    softmax.setInput(&dense2);
    //
    //    auto model = ml::Model<real_t, elsa::ml::MlBackend::Dnnl>(&input, &softmax);

    auto model = ml::Sequential(ml::Input(VolumeDescriptor{{28 * 28}}, 1),
                                ml::Dense(128, ml::Activation::Relu),
                                ml::Dense(10, ml::Activation::Relu), ml::Softmax());

    // Define an Adam optimizer
    auto opt = ml::Adam();

    // Compile the model
    model.compile(ml::SparseCategoricalCrossentropy(), &opt);

    std::vector<DataContainer<real_t>> trainingData = loadData("mnist_train.csv");
    printf("training data size: %zu\n", trainingData.size());

    std::vector<DataContainer<real_t>> trainingLabels = loadLabels("mnist_train.csv");
    printf("training labels size: %zu\n", trainingLabels.size());

    //    model.fit(trainingData, trainingLabels, 10);

    //    DataContainer<real_t> pred = model.predict(loadData("mnist_test.csv")[0]);
    //    printf("The net predicted: %f and the ground truth is: %f\n", pred[0],
    //           loadLabels("mnist_test.csv")[0][0]);
}

int main()
{
    try {
        mnist_example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
