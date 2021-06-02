#include "elsa.h"

#include <iostream>

using namespace elsa;

std::vector<DataContainer<real_t>> loadData(const std::string& fileName,
                                            unsigned int batchSize = 10)
{
    std::vector<DataContainer<real_t>> trainingImages;
    std::vector<DataContainer<real_t>> batchedImages;

    std::ifstream myfile;
    std::string line;
    myfile.open(fileName);
    if (myfile.is_open()) {
        printf("opened %s\n", fileName.c_str());
    }
    std::getline(myfile, line);
    //    int count = 0;
    while (!myfile.eof()) {
        //        if (count == 10)
        //            break;
        DataContainer<real_t> vectorizedImage(VolumeDescriptor{28 * 28});
        int pixelDensity = 0;
        int pixelIndex = 0;
        for (unsigned long i = 0; i < line.size(); i++) {
            if (i == 0) {
                continue;
            }
            char c = line[i];
            if (c == ',') {
                //                printf("num is %d\n", pixelDensity);
                vectorizedImage[pixelIndex++] = static_cast<real_t>(pixelDensity);
                pixelDensity = 0;
                continue;
            }
            pixelDensity = pixelDensity * 10 + (int) c - '0';
        }
        //        printf("data %f, %f, %f, %f, %f\n", vectorizedImage[0], vectorizedImage[1],
        //               vectorizedImage[2], vectorizedImage[3], vectorizedImage[4]);
        //        printf("vectorizedImage size: %lu\n", vectorizedImage.getSize());
        //        if (batchedImages.size() == batchSize) {
        //            DataContainer<real_t> ijgk(VolumeDescriptor{batchSize, 28 * 28});
        //            ijgk = 0;
        trainingImages.push_back(vectorizedImage);
        //            batchedImages.clear();
        //        } else {
        //            batchedImages.push_back(vectorizedImage);
        //        }
        std::getline(myfile, line);
        //        count++;
    }
    //    for (real_t pix : trainingImages[60000 - 1]) {
    //        printf("%f, ", pix);
    //    }
    myfile.close();

    //    DataContainer<real_t> ijgk(VolumeDescriptor{batchSize, 28 * 28});
    //    ijgk = 0;
    //    trainingImages.push_back(ijgk);
    //    trainingImages.push_back(ijgk);
    //    trainingImages.push_back(ijgk);
    //    //(3, 5, 784)
    return trainingImages;
}

std::vector<DataContainer<real_t>> loadLabels(const std::string& fileName,
                                              unsigned int batchSize = 10)
{
    std::vector<DataContainer<real_t>> trainingLabels;
    std::ifstream myfile;
    std::string line;
    myfile.open(fileName);
    if (myfile.is_open()) {
        printf("opened %s\n", fileName.c_str());
    }
    std::getline(myfile, line);
    //    int count = 0;
    while (!myfile.eof()) {
        //        if (count == 10)
        //            break;
        char c = line[0];
        DataContainer<real_t> imageLabel(VolumeDescriptor{1});
        imageLabel = static_cast<real_t>((int) c - '0');
        //        printf("%f\n", imageLabel[0]);
        trainingLabels.push_back(imageLabel);
        std::getline(myfile, line);
        //        count++;
    }
    //    printf("label: %f and label: %f and label: %f and label: %f and label: %f and label:
    //    %f\n",
    //           trainingLabels[60000 - 1][0], trainingLabels[60000 - 2][0], trainingLabels[60000 -
    //           3][0], trainingLabels[60000 - 4][0], trainingLabels[60000 - 5][0],
    //           trainingLabels[60000 - 6][0]);
    myfile.close();

    //    DataContainer<real_t> ijgk(VolumeDescriptor{batchSize, 1});
    //    ijgk = 0;
    //    trainingLabels.push_back(ijgk);
    //    trainingLabels.push_back(ijgk);
    //    trainingLabels.push_back(ijgk);
    //    //(3, 5, 1)
    return trainingLabels;
}

void mnist_example()
{
    //    VolumeDescriptor inputDescriptor{28 * 28, 1};
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
    //    auto model = ml::Model<real_t, elsa::ml::MlBackend::Cudnn>(&input, &softmax);

    auto model = ml::Sequential<real_t, elsa::ml::MlBackend::Cudnn>(
        ml::Input(VolumeDescriptor{28 * 28}, 1), ml::Dense(28 * 28, ml::Activation::Relu),
        ml::Dense(128, ml::Activation::Relu), ml::Dense(10, ml::Activation::Relu), ml::Softmax());

    // Define an Adam optimizer
    auto opt = ml::Adam(0.00004f);

    // Compile the model
    model.compile(ml::SparseCategoricalCrossentropy(), &opt);

    printf("after compiling the model\n");
    std::vector<DataContainer<real_t>> trainingData = loadData("mnist_train.csv", 5);
    printf("training data size: %zu\n", trainingData.size());

    std::vector<DataContainer<real_t>> trainingLabels = loadLabels("mnist_train.csv", 5);
    printf("training labels size: %zu\n", trainingLabels.size());

    model.fit(trainingData, trainingLabels, 10);
    printf("after training the model\n");

    DataContainer<real_t> pred = model.predict(loadData("mnist_train.csv")[0]);
    printf("The net predicted: ");
    for (real_t v : pred) {
        printf("%f, ", v);
    }
    printf("\n And the ground truth is: %f\n", loadLabels("mnist_train.csv")[0][0]);
}
//(60000, 1, 784)
//(12000, 5, 784)
int main()
{
    try {
        mnist_example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
