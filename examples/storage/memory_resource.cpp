/// Elsa example program: using the dedicated memory resources and scoped mr

#include "elsa.h"

#include <iostream>

using namespace elsa;

void exampleMemoryResources()
{
    auto printVec = [](const char* name, ContiguousStorage<int>& v) {
        std::cout << name << " [" << v.resource().get() << "] -> {";
        for (size_t i = 0; i < v.size(); ++i)
            std::cout << (i > 0 ? ", " : "") << v[i];
        std::cout << "}" << std::endl;
    };

    /* generate a contigous-vector, which uses the default resource */
    ContiguousStorage<int> x = {1, 2, 3, 4, 5};

    /* generate a contiguous-vector, which uses a different resource */
    ContiguousStorage<int> y = ContiguousStorage<int>({5, 6, 7, 8, 9}, mr::PoolResource::make());

    /* operations on the vectors */
    y.erase(y.begin() + 1, y.begin() + 2);
    x.insert(x.end(), y.begin(), y.end());

    {
        /* instantiating a scoped-mr */
        mr::hint::ScopedMR _scope{mr::CacheResource::make()};

        /* generate a contiguous-vectors, which uses the scoped-resource */
        ContiguousStorage<int> z0 = {10, 11, 12, 13, 14};
        ContiguousStorage<int> z1 = {15, 16, 17, 18, 19};

        /* copy the contents of y into z0, z0 will keep its resource */
        z0 = y;

        /* copy the content of z1 into y, y will use the incoming resource as its resource */
        y.assign(z1, z1.resource());

        /* x will contain the values of z0, but they will be copied, as the
         *  resource of x and z0 differ */
        x = std::move(z0);

        printVec("z0", z0);
        printVec("z1", z1);
    }

    /* example of cleaned-up scope */
    ContiguousStorage<int> z = {20, 21, 22, 23, 24};

    /*
     *   Final state:
     *      x uses global default-resource with values {5, 7, 8, 9}
     *      y uses the CacheResource of the scoped with values {15, 16, 17, 18, 19}
     *      z uses global default-resource with values {20, 21, 22, 23, 24}
     *
     *      z0 used scoped CacheResource and is empty at destruction
     *      z1 used scoped CacheResource and values {15, 16, 17, 18, 19} at destruction
     */
    printVec("x", x);
    printVec("y", y);
    printVec("z", z);
}

int main()
{
    try {
        exampleMemoryResources();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
