#include "Common.h"

#include <sstream>

namespace elsa
{
    namespace ml
    {
        namespace detail
        {

#define ELSA_ML_CASE_STATEMENT(T, X) \
    case T::X:                       \
        return #X

            std::string getEnumMemberAsString(LayerType type)
            {
                switch (type) {
                    ELSA_ML_CASE_STATEMENT(LayerType, Undefined);
                    ELSA_ML_CASE_STATEMENT(LayerType, Input);
                    ELSA_ML_CASE_STATEMENT(LayerType, Dense);
                    ELSA_ML_CASE_STATEMENT(LayerType, Activation);
                    ELSA_ML_CASE_STATEMENT(LayerType, Conv1D);
                    ELSA_ML_CASE_STATEMENT(LayerType, Conv2D);
                    ELSA_ML_CASE_STATEMENT(LayerType, Conv3D);
                    ELSA_ML_CASE_STATEMENT(LayerType, Conv2DTranspose);
                    ELSA_ML_CASE_STATEMENT(LayerType, Conv3DTranspose);
                    ELSA_ML_CASE_STATEMENT(LayerType, MaxPooling1D);
                    ELSA_ML_CASE_STATEMENT(LayerType, MaxPooling2D);
                    ELSA_ML_CASE_STATEMENT(LayerType, MaxPooling3D);
                    ELSA_ML_CASE_STATEMENT(LayerType, AveragePooling1D);
                    ELSA_ML_CASE_STATEMENT(LayerType, AveragePooling2D);
                    ELSA_ML_CASE_STATEMENT(LayerType, AveragePooling3D);
                    ELSA_ML_CASE_STATEMENT(LayerType, Sum);
                    ELSA_ML_CASE_STATEMENT(LayerType, Concatenate);
                    ELSA_ML_CASE_STATEMENT(LayerType, Reshape);
                    ELSA_ML_CASE_STATEMENT(LayerType, Flatten);
                    ELSA_ML_CASE_STATEMENT(LayerType, Softmax);
                    ELSA_ML_CASE_STATEMENT(LayerType, UpSampling1D);
                    ELSA_ML_CASE_STATEMENT(LayerType, UpSampling2D);
                    ELSA_ML_CASE_STATEMENT(LayerType, UpSampling3D);
                    ELSA_ML_CASE_STATEMENT(LayerType, Projector);
                    default:
                        return "unknown";
                }
            }

            std::string getEnumMemberAsString(Initializer initializer)
            {
                switch (initializer) {
                    ELSA_ML_CASE_STATEMENT(Initializer, Zeros);
                    ELSA_ML_CASE_STATEMENT(Initializer, Ones);
                    ELSA_ML_CASE_STATEMENT(Initializer, Uniform);
                    ELSA_ML_CASE_STATEMENT(Initializer, Normal);
                    ELSA_ML_CASE_STATEMENT(Initializer, GlorotUniform);
                    ELSA_ML_CASE_STATEMENT(Initializer, GlorotNormal);
                    ELSA_ML_CASE_STATEMENT(Initializer, HeUniform);
                    ELSA_ML_CASE_STATEMENT(Initializer, HeNormal);
                    ELSA_ML_CASE_STATEMENT(Initializer, RamLak);
                    default:
                        return "unknown";
                }
            }

            std::string getEnumMemberAsString(MlBackend backend)
            {
                switch (backend) {
                    ELSA_ML_CASE_STATEMENT(MlBackend, Dnnl);
                    ELSA_ML_CASE_STATEMENT(MlBackend, Cudnn);
                    default:
                        return "unknown";
                }
            }

            std::string getEnumMemberAsString(PropagationKind propagation)
            {
                switch (propagation) {
                    ELSA_ML_CASE_STATEMENT(PropagationKind, Forward);
                    ELSA_ML_CASE_STATEMENT(PropagationKind, Backward);
                    ELSA_ML_CASE_STATEMENT(PropagationKind, Full);
                    default:
                        return "unknown";
                }
            }
#undef ELSA_ML_CASE_STATEMENT
        } // namespace detail

    } // namespace ml
} // namespace elsa