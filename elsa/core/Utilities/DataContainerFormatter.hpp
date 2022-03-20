#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "FormatConfig.h"

#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace elsa
{

    /**
     * @brief Pretty printing for elsa::DataContainer.
     *
     * Inspired by numpy's formatter, supports n-dimensional string representation
     * of stored values in a `DataContainer`.
     *
     * @author Jonas Jelten - initial code
     *
     * @tparam data_t: datacontainer element type
     */
    template <typename data_t>
    class DataContainerFormatter
    {
    public:
        /**
         * Create a formatter with default config.
         */
        DataContainerFormatter() : config{this->get_default_format_config()} {}

        /**
         * Create a formatter with custom config.
         */
        DataContainerFormatter(const format_config& config) : config{config} {}

        /**
         * get the default formatting config.
         * the class defaults are adjusted due to the `data_t`.
         */
        format_config get_default_format_config()
        {
            format_config cfg;
            if constexpr (isComplex<data_t>) {
                cfg.summary_items /= 2;
            }

            return cfg;
        }

        /// stream a pretty format to is using the current configuration.
        void format(std::ostream& os, const DataContainer<data_t>& dc)
        {
            using namespace std::string_literals;

            auto dims = dc.getDataDescriptor().getNumberOfDimensions();
            auto shape = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension();
            os << "DataContainer<dims=" << dims << ", shape=(" << shape.transpose() << ")>"
               << std::endl;

            auto format_element = this->get_element_formatter<data_t>(dc);

            std::function<void(std::ostream & stream, index_t current_dim, IndexVector_t & index,
                               std::string_view indent_prefix)>
                recurser;
            recurser = [this, &recurser, &dc, &dims, &shape,
                        &format_element](std::ostream& os, index_t current_dim,
                                         IndexVector_t& index, std::string_view hanging_indent) {
                index_t dims_left = dims - current_dim;
                index_t next_dim = current_dim + 1;

                if (dims_left == 0) {
                    // get the actual element, recursion terminator! \o/
                    format_element(os, dc(index));
                    return;
                }

                std::string next_hanging_indent = std::string(hanging_indent) + " "s;

                index_t dim_size = shape(current_dim);
                auto backidx = [dim_size](index_t idx) { return dim_size - idx; };

                index_t leading_items, trailing_items;
                bool do_summary =
                    (this->config.summary_enabled and 2 * this->config.summary_items < dim_size);
                if (do_summary) {
                    leading_items = this->config.summary_items;
                    trailing_items = this->config.summary_items;
                } else {
                    leading_items = 0;
                    trailing_items = dim_size;
                }

                os << "[";

                // last dimension, i.e. the rows
                if (dims_left == 1) {

                    // TODO c++20 use views::iota
                    for (index_t i = 0; i < leading_items; i++) {
                        index(current_dim) = i;
                        recurser(os, next_dim, index, next_hanging_indent);
                        os << this->config.separator;
                    }

                    if (do_summary) {
                        os << this->config.summary_elem << this->config.separator;
                    }

                    for (index_t i = trailing_items; i > 1; i--) {
                        index(current_dim) = backidx(i);
                        recurser(os, next_dim, index, next_hanging_indent);
                        os << this->config.separator;
                    }

                    index(current_dim) = backidx(1);
                    recurser(os, next_dim, index, next_hanging_indent);
                } else {
                    // newlines between rows
                    // the more dimensions, the more newlines.
                    auto line_separator = (std::string(this->rstrip(this->config.separator))
                                           + std::string(dims_left - 1, '\n'));

                    for (index_t i = 0; i < leading_items; i++) {
                        index(current_dim) = i;
                        if (i != 0) {
                            os << hanging_indent;
                        }
                        recurser(os, next_dim, index, next_hanging_indent);
                        os << line_separator;
                    }

                    if (do_summary) {
                        os << hanging_indent << " " << this->config.summary_elem_vertical
                           << line_separator;
                    }

                    // remaining but the last element
                    for (index_t i = trailing_items; i > 1; i--) {
                        index(current_dim) = backidx(i);
                        if (do_summary or i != trailing_items) {
                            os << hanging_indent;
                        }
                        recurser(os, next_dim, index, next_hanging_indent);
                        os << line_separator;
                    }

                    // indent for the current dim's last entry
                    // we skip it if it's the only entry
                    if (trailing_items > 1) {
                        os << hanging_indent;
                    }

                    // print the last element
                    index(current_dim) = backidx(1);
                    recurser(os, next_dim, index, next_hanging_indent);
                }

                os << "]";
            };

            // we'll modify this index on the fly when we walk over the datacontainer
            IndexVector_t dc_index{dims};
            auto prefix = " "s; // skip over initial [
            recurser(os, 0, dc_index, prefix);
        }

    private:
        /**
         * adjust a given string_view and remove all whitespace from the right side.
         */
        std::string_view rstrip(std::string_view text)
        {
            auto end_it = std::rbegin(text);
            while (end_it != std::rend(text) and *end_it == ' ') {
                ++end_it;
            }

            // TODO c++20: use stringview iterator constructor directly
            const char* start = &(*std::begin(text));
            const char* end = &(*end_it) + 1;

            ssize_t len = end - start;
            if (len < 0) {
                throw elsa::InternalError{"rstrip length sanity check failed"};
            }
            return std::string_view(start, static_cast<size_t>(len));
        }

        /**
         * Given a DataContainer, generate a function that will format one element beautifully.
         * This element is padded to the max width of the other elements.
         *
         * @param dc: data to get the formatter for.
         *
         * TODO add new features:
         * unique: display elements in such a way they are uniquely distinguishable
         *         (dynamically adjust precision)
         * precision: if unique is false, use this float precision.
         */
        template <typename T>
        std::function<std::ostream&(std::ostream& os, const T& elem)>
            get_element_formatter(const DataContainer<T>& dc)
        {
            if constexpr (elsa::isComplex<T>) {
                // format both components independently
                auto real_formatter = get_element_formatter(DataContainer<T>{real(dc)});
                auto imag_formatter = get_element_formatter(DataContainer<T>{imag(dc)});

                return [real_formatter, imag_formatter](std::ostream & os, const T& elem) -> auto&
                {
                    real_formatter(os, elem.real());
                    os << "+";
                    imag_formatter(os, elem.imag());
                    os << "j";
                    return os;
                };
            } else if constexpr (std::is_floating_point_v<T>) {
                // TODO: handle non-finite (not nan, inf) elements
                // TODO: align stuff at the . and do right and left padding

                bool suppress_small = true;
                bool use_exp = false;

                T val_max = dc.maxElement();
                T val_min = dc.minElement();

                if (val_max > 1e7
                    or (not suppress_small and (val_min < 0.0001 or val_max / val_min > 1000.0))) {

                    use_exp = true;
                }

                // TODO: possible optimization - no underlying string needed,
                // could be a /dev/null-like storage,
                // since we only use tellp.
                std::ostringstream teststream;
                if (use_exp) {
                    teststream << std::scientific;
                } else {
                    teststream << std::defaultfloat;
                }

                // setw wants int...
                int maxlen = 0;
                for (elsa::index_t idx = 0; idx < dc.getSize(); ++idx) {
                    teststream.str("");
                    teststream.clear();

                    auto&& elem = config.suppress_close_to_zero
                                          && std::abs(dc[idx]) < config.suppression_epsilon
                                      ? static_cast<data_t>(0)
                                      : dc[idx];

                    teststream << elem;
                    auto len = static_cast<int>(teststream.tellp());

                    if (len > maxlen) {
                        maxlen = len;
                    }
                }

                auto streamflags = teststream.flags();

                return [
                    maxlen, streamflags, do_suppress = config.suppress_close_to_zero,
                    eps = config.suppression_epsilon
                ](std::ostream & os, const T& elem) -> auto&
                {
                    os.flags(streamflags);
                    os << std::setw(maxlen);
                    os << (do_suppress && std::abs(elem) < eps ? static_cast<data_t>(0) : elem);
                    return os;
                };
            } else {
                // setw wants int, string::size returns size_t. great.
                int maxlen = static_cast<int>(std::max(std::to_string(dc.maxElement()).size(),
                                                       std::to_string(dc.minElement()).size()));
                return [maxlen](std::ostream & os, const T& elem) -> auto&
                {
                    auto&& elem_str = std::to_string(elem);
                    os << std::setw(maxlen);
                    os << std::to_string(elem);
                    return os;
                };
            }
        }

        /// formatting output configuration
        format_config config;
    };

} // namespace elsa
