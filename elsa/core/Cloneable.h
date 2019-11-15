#pragma once

#include <memory>

namespace elsa
{
    /**
     * \brief Class implementing polymorphic clones with smart pointers and CRTP, as well as
     * comparison operators.
     *
     * \author Tobias Lasser
     *
     * This class provides a clone method using CRTP to support covariance with smart pointers.
     * For details see
     * https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/.
     */
    template <typename Derived>
    class Cloneable
    {
    public:
        /// default constructor
        Cloneable() = default;
        /// virtual default destructor
        virtual ~Cloneable() = default;

        /// clone object, returning an owning unique_ptr
        std::unique_ptr<Derived> clone() const { return std::unique_ptr<Derived>(cloneImpl()); }

        /// comparison operators
        bool operator==(const Derived& other) const
        {
            return typeid(*this) == typeid(other) && isEqual(other);
        }

        bool operator!=(const Derived& other) const { return !(*this == other); }

    protected:
        /// actual clone implementation method, abstract to force override in derived classes
        virtual Derived* cloneImpl() const = 0;

        /// actual comparison method, abstract to force override in derived classes
        virtual bool isEqual(const Derived& other) const = 0;

        /// default copy constructor, protected to not be publicly available (but available for
        /// cloneImpl)
        Cloneable(const Cloneable&) = default;
        /// default copy assignment, protected to not be publicly available
        Cloneable& operator=(const Cloneable&) = default;
    };

} // namespace elsa
