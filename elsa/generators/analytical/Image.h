#include "Cloneable.h"
#include "CurvedDetectorDescriptor.h"
#include "DataDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include <Eigen/src/Core/Matrix.h>

#include <iostream>
#include <memory>
#include <unordered_set>
#include <concepts>

namespace elsa::phantoms
{

    template <typename data_t>
    using Position = Eigen::Vector2<data_t>;

    template <typename data_t>
    class Image : public Cloneable<Image<data_t>>
    {
    public:
        virtual DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor)
        {
            assert(is<CurvedDetectorDescriptor>(sinogramDescriptor));
            assert(sinogramDescriptor.getNumberOfDimensions() == 2);
            return DataContainer<data_t>{sinogramDescriptor};
        }

    protected:
        virtual Image<data_t>* cloneImpl() const = 0;
        virtual bool isEqual(const Image<data_t>& other) const = 0;
    };

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
            assert(is<PlanarDetectorDescriptor>(sinogramDescriptor));
            assert(sinogramDescriptor.getNumberOfDimensions() == 2);

            DataContainer<data_t> sinogram{*sinogramDescriptor.clone()};
            sinogram += a->makeSinogram(sinogramDescriptor);
            sinogram += b->makeSinogram(sinogramDescriptor);
            return sinogram;
        }

        std::unique_ptr<Image<data_t>> a, b;

        virtual Image<data_t>* cloneImpl() const override
        {
            return new Sum<data_t>{a->clone(), b->clone()};
        }
        virtual bool isEqual(const Image<data_t>&) const override { return false; };

    private:
    };

    template <typename data_t>
    Sum<data_t> operator+(const Image<data_t>& a, const Image<data_t>& b)
    {
        return Sum{a.clone(), b.clone()};
    }

    template <typename data_t = float>
    class Ellipse : public Image<data_t>
    {
    public:
        Ellipse(Position<data_t> center, data_t a, data_t b) : center{center}, a{a}, b{b} {}

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) override
        {
            std::cout << "Ellipse ()" << std::endl;
            return DataContainer<data_t>{sinogramDescriptor};
        }

    protected:
        virtual Ellipse<data_t>* cloneImpl() const override { return new Ellipse(*this); }
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            (void) other;
            return false;
        };

    private:
        Position<data_t> center;
        data_t a, b;
    };

} // namespace elsa::phantoms