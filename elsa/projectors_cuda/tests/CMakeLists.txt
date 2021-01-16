# enable ctest and Catch test discovery
include(CTest)
include(Catch)

ELSA_TEST(SiddonsMethodCUDA)
ELSA_TEST(JosephsMethodCUDA)
