#include "elsa.h"

#include <iostream>
#include <models/AutoEncoder.h>

using namespace elsa;

void dl_models_example()
{
    // use a predefined model
    elsa::ml::AutoEncoder<real_t, elsa::ml::MlBackend::Cudnn> model(1, 1);

    // TODO ideally also something like
    // elsa::ml::AutoEncoder<real_t, elsa::ml::MlBackend::Cudnn> model(1, 1, true);
    // TODO true here would refer to a preTrained flag, this would allow us to skip training

    // define an Adam optimizer
    auto opt = ml::Adam(0.0008f);

    // compile the model
    model.compile(ml::MeanSquaredError(), &opt);

    DataContainer noise(VolumeDescriptor{28, 28});
    EDF::write(noise, "mystery_noise.edf");

    // train the predefined model
    model.fit({noise}, {noise}, 4);

    // use the model to make predictions
    DataContainer<real_t> pred = model.predict(noise);
    EDF::write(pred, "mystery_pred.edf");
}

int main()
{
    try {
        dl_models_example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
