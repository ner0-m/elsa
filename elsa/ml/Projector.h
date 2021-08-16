#pragma once

#include "elsaDefines.h"
#include "Layer.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"

namespace elsa::ml
{
    template <typename data_t = real_t>
    class Projector : public Layer<data_t>
    {
    public:
        Projector(LinearOperator<data_t>* op, const std::string& name = "")
            : Layer<data_t>(LayerType::Projector, name), operator_(op)
        {
        }

        void computeOutputDescriptor() override
        {
            auto& dd = operator_->getDomainDescriptor();
            IndexVector_t outDims(dd.getNumberOfDimensions() + 1);
            outDims.head(dd.getNumberOfDimensions()) = dd.getNumberOfCoefficientsPerDimension();
            outDims[dd.getNumberOfDimensions()] =
                this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension().tail(1)(0);
            this->outputDescriptor_ = VolumeDescriptor(outDims).clone();
        }

        const LinearOperator<data_t>& getLinearOperator() const { return *operator_; }

        LinearOperator<data_t>& getLinearOperator() { return *operator_; }

    private:
        LinearOperator<data_t>* operator_;
    };
} // namespace elsa::ml