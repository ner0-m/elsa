#include "ProjectionOperator.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ProjectionOperator<data_t>::ProjectionOperator(const DataDescriptor& descriptor)
        : LinearOperator<data_t>(descriptor, descriptor)
    {
    }

    template <typename data_t>
    void ProjectionOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                               DataContainer<data_t>& Px) const
    {
        // TODO add logic
        Timer timeguard("ProjectionOperator", "apply");
        Px = x;
    }

    template <typename data_t>
    void ProjectionOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                      DataContainer<data_t>& Pty) const
    {
        // TODO add logic
        Timer timeguard("ProjectionOperator", "applyAdjoint");
        Pty = y;
    }

    template <typename data_t>
    ProjectionOperator<data_t>* ProjectionOperator<data_t>::cloneImpl() const
    {
        return new ProjectionOperator(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool ProjectionOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherProjector = dynamic_cast<const ProjectionOperator*>(&other);
        return static_cast<bool>(otherProjector);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProjectionOperator<float>;
    template class ProjectionOperator<std::complex<float>>;
    template class ProjectionOperator<double>;
    template class ProjectionOperator<std::complex<double>>;
} // namespace elsa
