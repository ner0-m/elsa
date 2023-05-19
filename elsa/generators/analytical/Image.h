#pragma once
#include "Cloneable.h"
#include "DataDescriptor.h"
#include "DetectorDescriptor.h"
#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include "DataContainer.h"

#include <memory>

namespace elsa::phantoms
{

    template <typename data_t>
    class Image : public Cloneable<Image<data_t>>
    {
    public:
        virtual DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) = 0;

        virtual Image<data_t>* cloneImpl() const = 0;

    protected:
        virtual bool isEqual(const Image<data_t>& other) const = 0;
    };

} // namespace elsa::phantoms
