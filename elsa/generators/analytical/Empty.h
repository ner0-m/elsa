#pragma once
#include "DataContainer.h"
#include "Image.h"
#include "TypeCasts.hpp"
#include <memory>
#include <unordered_set>

namespace elsa::phantoms
{

    template <typename data_t = float>
    class Canvas : public Image<data_t>
    {
    public:
        Canvas() = default;

        explicit Canvas(const std::unordered_set<std::unique_ptr<Image<data_t>>>& components)
        {
            for (const auto& c : components) {
                this->components.insert(c->clone());
            }
        }

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) override
        {
            DataContainer<data_t> sinogram{sinogramDescriptor};
            for (const auto& component : components) {
                sinogram += component->makeSinogram(sinogramDescriptor);
            }
            return sinogram;
        }
        Canvas& operator+=(const Image<data_t>& image)
        {
            components.insert(image.clone());
            return *this;
        }

        Canvas& operator-=(const Image<data_t>& image)
        {
            components.insert((-1 * image).clone());
            return *this;
        }

    protected:
        virtual Canvas<data_t>* cloneImpl() const override { return new Canvas{components}; }
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            if (!is<Canvas<data_t>>(other))
                return false;
            const Canvas<data_t>& asCanvas = downcast<Canvas<data_t>>(other);

            auto compareUnique = [](const std::unique_ptr<Image<data_t>>& a,
                                    const std::unique_ptr<Image<data_t>>& b) { return *a == *b; };

            return std::equal(components.begin(), components.end(), asCanvas.components.begin(),
                              compareUnique);
        };

    private:
        std::unordered_set<std::unique_ptr<Image<data_t>>> components;
    };

} // namespace elsa::phantoms