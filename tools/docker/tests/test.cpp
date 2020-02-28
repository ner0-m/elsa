#include <iostream>
#include <functional>

int main()
{
    []() { std::cout << "Hello World, C++11\n"; }();
    return 0;
}
