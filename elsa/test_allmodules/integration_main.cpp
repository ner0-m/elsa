#include "elsa.h"
#include "TestDriver.h"

using namespace elsa;

int main()
{
#ifdef ELSA_CUDA_PROJECTORS
    benchDriver<CG, SiddonsMethodCUDA>(2, 256, 20);
    benchDriver<CG, JosephsMethodCUDA>(2, 256, 20);
    benchDriver<ADMM, CG, SoftThresholding, SiddonsMethodCUDA>(2, 128, 5);
#endif

    benchDriver<CG, SiddonsMethod>(2, 64, 5);
    benchDriver<ADMM, CG, SoftThresholding, SiddonsMethod>(2, 32, 5);
    benchDriver<ISTA, SiddonsMethod>(2, 32, 5);
}
