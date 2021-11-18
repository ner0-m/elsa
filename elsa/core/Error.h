#pragma once

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

namespace elsa
{

    namespace detail
    {
        /**
         * @brief Base exception/error class relying on backward-cpp to retrieve the stacktrace on
         * error
         */
        class BaseError : public std::runtime_error
        {
        public:
            BaseError(std::string msg);
        };

        std::ostream& operator<<(std::ostream& os, const BaseError& err);

    } // namespace detail

    class Error : public detail::BaseError
    {
    public:
        Error(const std::string& msg);
    };

    /**
     * Internal Error, thrown when some interal sanity check failed.
     */
    class InternalError : public detail::BaseError
    {
    public:
        InternalError(const std::string& msg);
    };

    /**
     * Faulty logic due to violation of expected preconditions.
     */
    class LogicError : public detail::BaseError
    {
    public:
        LogicError(const std::string& msg);
    };

    /**
     * User gave something wrong to our function.
     */
    class InvalidArgumentError : public detail::BaseError
    {
    public:
        InvalidArgumentError(const std::string& msg);
    };

    /**
     * Could not cast to the given type
     */
    class BadCastError : public detail::BaseError
    {
    public:
        BadCastError(const std::string& msg);
    };

} // namespace elsa
