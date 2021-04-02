#include "spdlog/fmt/ostr.h"

std::ostream& operator<<(std::ostream& os, const elsa::BlockDescriptor& desc)
{
    return os << "{ BlockDescriptor { " << desc.getNumberOfBlocks() << " blocks"
              << " }}";
}
