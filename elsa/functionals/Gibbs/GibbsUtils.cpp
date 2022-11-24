#include "GibbsUtils.h"
#include "Error.h"

namespace elsa
{
    namespace Gibbs
    {

        IndexVector_t oneDimension(index_t val)
        {
            IndexVector_t res(1);
            res << val;
            return res;
        };

        std::vector<IndexVector_t> generateNeighbourShift(index_t n)
        {
            if (n <= 0)
                return {};
            if (n == 1)
                return {oneDimension(-1), oneDimension(0), oneDimension(1)};
            std::vector<IndexVector_t> prev = generateNeighbourShift(n - 1);
            std::vector<IndexVector_t> res(3 * prev.size());
            auto pos = res.begin();
            for (index_t shift = -1; shift <= 1; shift++) {
                for (IndexVector_t prevShift : prev) {
                    IndexVector_t next(n);
                    auto nextp = next.begin();
                    *nextp++ = shift;
                    std::copy(prevShift.begin(), prevShift.end(), nextp);
                    *pos = next;
                    pos++;
                }
            }
            return res;
        }

    } // namespace Gibbs
} // namespace elsa