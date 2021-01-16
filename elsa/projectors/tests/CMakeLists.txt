# enable ctest and Catch test discovery
include(CTest)
include(Catch)


# the actual tests
ELSA_TEST(BoundingBox)
ELSA_TEST(Intersection)
ELSA_TEST(TraverseAABB)
ELSA_TEST(BinaryMethod)
ELSA_TEST(SiddonsMethod)
ELSA_TEST(JosephsMethod)
