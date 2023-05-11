#pragma once
#include "Image.h"
#include "Scaling.h"

namespace elsa::phantoms
{

    template <typename data_t>
    class Sum : public Image<data_t>
    {
    public:
        Sum(std::unique_ptr<Image<data_t>> a, std::unique_ptr<Image<data_t>> b)
            : a{std::move(a)}, b{std::move(b)}
        {
        }

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) override
        {
            DataContainer<data_t> sinogram{*sinogramDescriptor.clone()};
            sinogram += a->makeSinogram(sinogramDescriptor);
            sinogram += b->makeSinogram(sinogramDescriptor);
            return sinogram;
        }

        virtual Image<data_t>* cloneImpl() const override
        {
            return new Sum<data_t>{a->clone(), b->clone()};
        }

    protected:
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            if (!is<Sum<data_t>>(other))
                return false;

            const auto& asSum = downcast<Sum<data_t>>(other);
            return *a == *asSum.a && *b == *asSum.b;
        }

    private:
        std::unique_ptr<Image<data_t>> a, b;
    };

    template <typename data_t>
    Sum<data_t> operator+(const Image<data_t>& a, const Image<data_t>& b)
    {
        return Sum{a.clone(), b.clone()};
    }

    template <typename data_t>
    Sum<data_t> operator-(const Image<data_t>& a, const Image<data_t>& b)
    {
        return Sum{a.clone(), (-1 * b).clone()};
    }

} // namespace elsa::phantoms