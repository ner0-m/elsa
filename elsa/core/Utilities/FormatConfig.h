#pragma once

#include "elsaDefines.h"

#include <string>

namespace elsa
{
    /**
     * Formatting output configuration.
     */
    struct format_config {
        /// for too many elements, abbreviate the output
        bool summary_enabled = true;

        /// number of summary items to display - this also triggers the summary enabling
        /// if theres more than 2 * summary_items elements.
        index_t summary_items = 6;

        /// what is inserted between the summary items
        std::string summary_elem = "...";

        /// what is inserted vertically between summary items
        std::string summary_elem_vertical = "\u22EE";

        /// what's inserted between elements and newlines
        std::string separator = ", ";

        /// if a value is smaller than some epsilon (`suppression_epsilon`), just print 0
        /// instead
        bool suppress_close_to_zero = false;

        /// epsilon value for suppressing small numbers
        real_t suppression_epsilon = static_cast<real_t>(0.0000001);
    };
} // namespace elsa
