#pragma once
#include "Image.h"

namespace elsa::phantoms
{

    template <typename data_t, typename scalar = data_t>
    class Scaling : public Image<data_t>
    {
    public:
        Scaling(scalar k, std::unique_ptr<Image<data_t>> a) : k{k}, a{std::move(a)} {}

        data_t traceRay(const Ray_t<data_t>& ray) override { return k * a->traceRay(ray); }

    protected:
        virtual Image<data_t>* cloneImpl() const override
        {
            return new Scaling<data_t, scalar>{k, a->clone()};
        }
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            if (!is<Scaling<data_t, scalar>>(other))
                return false;

            const auto& asScaling = downcast<Scaling<data_t, scalar>>(other);
            return k == asScaling.k && *a == *asScaling.a;
        }

    private:
        scalar k;
        std::unique_ptr<Image<data_t>> a;
    };

    template <typename data_t, typename scalar = data_t>
    Scaling<data_t, scalar> operator*(scalar k, const Image<data_t>& a)
    {
        return Scaling{k, a.clone()};
    }
    template <typename data_t, typename scalar = data_t>
    Scaling<data_t, scalar> operator*(const Image<data_t>& a, scalar k)
    {
        return k * a;
    }

} // namespace elsa::phantoms