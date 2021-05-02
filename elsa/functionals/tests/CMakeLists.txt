# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(LinearResidual)
ELSA_TEST(L0PseudoNorm)
ELSA_TEST(L1Norm)
ELSA_TEST(L2NormPow2)
ELSA_TEST(WeightedL2NormPow2)
ELSA_TEST(LInfNorm)
ELSA_TEST(Huber)
ELSA_TEST(PseudoHuber)
ELSA_TEST(Quadric)
ELSA_TEST(EmissionLogLikelihood)
ELSA_TEST(TransmissionLogLikelihood)
ELSA_TEST(Constraint)
ELSA_TEST(WeightedL1Norm)
ELSA_TEST(BumpFunction)
