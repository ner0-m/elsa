#include "doctest/doctest.h"

#include "memory_resource/ContiguousVector.h"
#include "memory_resource/HostStandardResource.h"
#include "Assertions.h"

#include <map>
#include <set>
#include <vector>
#include <list>

using namespace elsa::mr;

struct TestStatsCounter {
    bool intest = false;
    MemoryResource resource;

    /* for complex type */
    std::set<const struct ComplexType*> preTestConstructed;
    std::set<const struct ComplexType*> constructed;
    size_t invalidDestruct = 0;
    size_t doubleConstruct = 0;
    size_t invalidAssign = 0;
    size_t invalidAccess = 0;
    size_t destructCount = 0;

    /* uninitialized type */
    size_t defaultConstruct = 0;

    /* for allocator */
    std::map<void*, std::pair<size_t, size_t>> preTestAllocated;
    std::map<void*, std::pair<size_t, size_t>> allocations;
    size_t invalidFree = 0;
    size_t memoryOperations = 0;
    size_t allocOperations = 0;
};
static TestStatsCounter testStats;

class CheckedResource : private HostStandardResource
{
private:
    static constexpr uint8_t InitByteValue = 42;

public:
    template <class T>
    static constexpr T initValue()
    {
        T t = 0;
        for (size_t i = 0; i < sizeof(T); ++i)
            t |= static_cast<T>(InitByteValue) << (i * 8);
        return t;
    }

private:
    void* allocate(size_t size, size_t alignment) override
    {
        void* p = HostStandardResource::allocate(size, alignment);
        if (!testStats.intest) {
            testStats.preTestAllocated.insert({p, {size, alignment}});
            return p;
        }
        testStats.allocations.insert({p, {size, alignment}});
        ++testStats.allocOperations;

        /* fill the content with non-zero values */
        uint8_t* p8 = reinterpret_cast<uint8_t*>(p);
        for (size_t i = 0; i < size; i++)
            p8[i] = InitByteValue;
        return p;
    }
    void deallocate(void* ptr, size_t size, size_t alignment) noexcept override
    {
        if (!testStats.intest) {
            auto it = testStats.preTestAllocated.find(ptr);
            if (it != testStats.preTestAllocated.end()) {
                size = it->second.first;
                alignment = it->second.second;
                testStats.preTestAllocated.erase(it);
                HostStandardResource::deallocate(ptr, size, alignment);
            }
            return;
        }

        auto it = testStats.allocations.find(ptr);
        if (it == testStats.allocations.end() || it->second.first != size
            || it->second.second != alignment)
            ++testStats.invalidFree;
        else {
            testStats.allocations.erase(it);
            HostStandardResource::deallocate(ptr, size, alignment);
        }
    }
    void copyMemory(void* ptr, const void* src, size_t size) override
    {
        ++testStats.memoryOperations;
        HostStandardResource::copyMemory(ptr, src, size);
    }
    void moveMemory(void* ptr, const void* src, size_t size) override
    {
        ++testStats.memoryOperations;
        HostStandardResource::moveMemory(ptr, src, size);
    }
    void setMemory(void* ptr, const void* src, size_t stride, size_t count) override
    {
        ++testStats.memoryOperations;
        HostStandardResource::setMemory(ptr, src, stride, count);
    }

public:
    static MemoryResource make() { return MemoryResource::MakeRef(new CheckedResource()); }
};
struct ComplexType {
public:
    using tag = type_tags::complex;
    template <class T>
    static constexpr bool is = std::is_same<T, ComplexType>::value;
    static constexpr int initValue = 0;

private:
    int _payload = 0;

private:
    void _checkConstruct()
    {
        if (testStats.constructed.count(this) > 0)
            ++testStats.doubleConstruct;
        else if (testStats.intest)
            testStats.constructed.insert(this);
        else
            testStats.preTestConstructed.insert(this);
    }
    void _checkValid(const ComplexType& t, bool access) const
    {
        if (testStats.constructed.count(&t) == 0 && testStats.preTestConstructed.count(&t) == 0)
            ++(access ? testStats.invalidAccess : testStats.invalidAssign);
    }

public:
    ComplexType() { _checkConstruct(); }
    ComplexType(int i) : _payload(i) { _checkConstruct(); }
    ComplexType(const ComplexType& t) : _payload(t._payload)
    {
        _checkConstruct();
        _checkValid(t, true);
    }
    ComplexType(ComplexType&& t) noexcept : _payload(t._payload)
    {
        _checkConstruct();
        _checkValid(t, true);
    }
    ~ComplexType()
    {
        if (!testStats.intest) {
            testStats.preTestConstructed.erase(this);
            return;
        }

        auto it = testStats.constructed.find(this);
        if (it == testStats.constructed.end())
            ++testStats.invalidDestruct;
        else
            testStats.constructed.erase(it);
    }
    ComplexType& operator=(const ComplexType& t)
    {
        _checkValid(t, true);
        _checkValid(*this, false);
        _payload = t._payload;
        return *this;
    }
    ComplexType& operator=(ComplexType&& t)
    {
        _checkValid(t, true);
        _checkValid(*this, false);
        _payload = t._payload;
        return *this;
    }

    operator int() const
    {
        _checkValid(*this, true);
        return _payload;
    }
};
struct TrivialType {
public:
    using tag = type_tags::trivial;
    template <class T>
    static constexpr bool is = std::is_same<T, TrivialType>::value;
    static constexpr int initValue = 0;

private:
    int _payload = 0;

public:
    TrivialType() = default;
    TrivialType(int i) : _payload(i) {}
    ~TrivialType() { ++testStats.destructCount; }
    TrivialType(const TrivialType&) = default;
    TrivialType(TrivialType&&) = default;
    TrivialType& operator=(const TrivialType&) = default;
    TrivialType& operator=(TrivialType&&) = default;
    operator int() const { return _payload; }
};
struct UninitType : public TrivialType {
public:
    using tag = type_tags::uninitialized;
    template <class T>
    static constexpr bool is = std::is_same<T, UninitType>::value;
    static constexpr int initValue = CheckedResource::initValue<int>();

public:
    UninitType() { ++testStats.defaultConstruct; }
    UninitType(int i) : TrivialType(i) {}
    UninitType(const UninitType& t) = default;
    UninitType(UninitType&&) = default;
    UninitType& operator=(const UninitType&) = default;
    UninitType& operator=(UninitType&&) = default;
};

/* start measuring the stats / stop & verify */
static void StartTestStats()
{
    testStats.constructed.clear();
    testStats.allocations.clear();

    testStats.invalidDestruct = 0;
    testStats.doubleConstruct = 0;
    testStats.invalidAssign = 0;
    testStats.invalidAccess = 0;
    testStats.destructCount = 0;
    testStats.defaultConstruct = 0;
    testStats.invalidFree = 0;
    testStats.memoryOperations = 0;
    testStats.allocOperations = 0;

    testStats.intest = true;
}
static void VerifyTestStats(bool memOp)
{
    testStats.intest = false;

    CHECK_MESSAGE(testStats.constructed.empty(), "Not all objects were destructed");
    CHECK_MESSAGE(testStats.invalidDestruct == 0, "Uninitialized objects were destructed");
    CHECK_MESSAGE(testStats.doubleConstruct == 0, "Initialized objects were re-initialized");
    CHECK_MESSAGE(testStats.invalidAssign == 0, "Uninitialized objects were assigned");
    CHECK_MESSAGE(testStats.invalidAccess == 0, "Uninitialized objects were accessed");
    CHECK_MESSAGE(testStats.destructCount == 0, "Unexpected destructs occurred");
    if (memOp)
        CHECK_MESSAGE(testStats.memoryOperations > 0, "Expected memory operation did not occur");
    else
        CHECK_MESSAGE(testStats.memoryOperations == 0, "Unexpected memory operation occurred");
    CHECK_MESSAGE(testStats.defaultConstruct == 0,
                  "Uninitialized objects were default-constructed");
    CHECK_MESSAGE(testStats.allocations.empty(), "Memory leak detected");
    CHECK_MESSAGE(testStats.invalidFree == 0, "Invalid address was deallocated");

    /* reset all missed allocations */
    for (auto& alloc : testStats.allocations)
        testStats.resource->deallocate(alloc.first, alloc.second.first, alloc.second.second);
}

/* count how many values in the container match the given requirements */
template <class T>
static size_t CheckAllEqual(const T& t, int val)
{
    size_t count = 0;
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] == val)
            ++count;
    }
    return count;
}
template <class T, class It>
static size_t CheckAllMatch(const T& t, It it)
{
    size_t count = 0;
    auto tt = t.begin();
    while (tt != t.end()) {
        if (*tt == *it)
            ++count;
        ++tt;
        ++it;
    }
    return count;
}

TEST_SUITE_BEGIN("memoryresources");
TYPE_TO_STRING(ComplexType);
TYPE_TO_STRING(TrivialType);
TYPE_TO_STRING(UninitType);

TEST_CASE_TEMPLATE("ContiguousVector::Constructors", T, ComplexType, TrivialType, UninitType)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag>;

    SUBCASE("Default constructor")
    {
        StartTestStats();

        {
            Vector storage(mres);
            CHECK(storage.size() == 0);
        }

        VerifyTestStats(false);
    }

    int intCount0 = 50;
    SUBCASE("Constructor(size_t)")
    {
        StartTestStats();

        {
            Vector storage(intCount0, mres);
            REQUIRE(storage.size() == intCount0);
            CHECK(CheckAllEqual(storage, T::initValue) == intCount0);
        }

        VerifyTestStats(TrivialType::is<T>);
    }

    int intNum1 = 32, intCount1 = 17;
    SUBCASE("Constructor(size_t, const value&)")
    {
        T temp(intNum1);

        StartTestStats();

        {
            Vector storage(intCount1, temp, mres);
            REQUIRE(storage.size() == intCount1);
            CHECK(CheckAllEqual(storage, intNum1) == intCount1);
        }

        VerifyTestStats(!ComplexType::is<T>);
    }

    std::vector<int> intVec0 = {5, 9, 123, 130, -123, 2394, 591, 1203, 523, 123};
    SUBCASE("Constructor(ItType, ItType) [consecutive]")
    {
        StartTestStats();

        {
            Vector storage(intVec0.begin(), intVec0.end(), mres);
            REQUIRE(storage.size() == intVec0.size());
            CHECK(CheckAllMatch(storage, intVec0.begin()) == intVec0.size());
        }

        VerifyTestStats(!ComplexType::is<T>);
    }

    std::list<int> intList = {6, 123, -123, 504, 12321, 889123, 345, -543, -8123, 10148};
    SUBCASE("Constructor(ItType, ItType) [not-consecutive]")
    {
        StartTestStats();

        {
            Vector storage(intList.begin(), intList.end(), mres);
            REQUIRE(storage.size() == intList.size());
            CHECK(CheckAllMatch(storage, intList.begin()) == intList.size());
        }

        VerifyTestStats(false);
    }

    SUBCASE("Constructor(initializer_list) [implicit: consecutive]")
    {
        std::initializer_list<T> intInit = {1, -12, 0, -1123, 0532, 7123, -1239, -67345, 012, -4};

        StartTestStats();

        {
            Vector storage(intInit, mres);
            REQUIRE(storage.size() == intInit.size());
            CHECK(CheckAllMatch(storage, intInit.begin()) == intInit.size());
        }

        VerifyTestStats(!ComplexType::is<T>);
    }

    SUBCASE("Constructor(const self_type&) [implicit: consecutive]")
    {
        Vector intSelf({-1, 0, 123, 5, -123, 9348, -239411, -123, 64223, -123}, mres);

        StartTestStats();

        {
            Vector storage(intSelf, mres);
            REQUIRE(storage.size() == intSelf.size());
            CHECK(CheckAllMatch(storage, intSelf.begin()) == intSelf.size());
        }

        VerifyTestStats(!ComplexType::is<T>);
    }

    SUBCASE("Constructor(self_type&&) [implicit: consecutive]")
    {
        Vector intSelf(intList.begin(), intList.end(), mres);

        StartTestStats();

        {
            Vector storage(std::move(intSelf));
            // NOLINTNEXTLINE(*-use-after-move)
            CHECK(intSelf.size() == 0);
            REQUIRE(storage.size() == intList.size());
            CHECK(CheckAllMatch(storage, intList.begin()) == intList.size());

            /* necessary as move-constructor 'steals' the values from the pre-constructed storage */
            std::swap(testStats.preTestConstructed, testStats.constructed);
            std::swap(testStats.preTestAllocated, testStats.allocations);
        }

        VerifyTestStats(false);
    }

    testStats.resource.release();
}

TEST_SUITE_END();

/* test that ptr is != to zero on allocations and else zero */
/* test first capacity and grow of it */
/* test when resource is changed */

/* add test for 'uninitialization' */
/* add test for 'trivial' */
/* add test for 'exception-handling' */