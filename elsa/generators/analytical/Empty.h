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

        void addSinogram(const DataDescriptor& sinogramDescriptor,
                         const std::vector<Ray_t<data_t>>& rays,
                         DataContainer<data_t>& container) override
        {
            for (const auto& component : components) {
                component->addSinogram(sinogramDescriptor, rays, container);
            }
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