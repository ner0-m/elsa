#include "doctest/doctest.h"

#include "memory_resource/ContiguousVector.h"
#include "memory_resource/ContiguousWrapper.h"
#include "memory_resource/HostStandardResource.h"
#include "Assertions.h"

#include <map>
#include <set>
#include <vector>
#include <list>
#include <random>

using namespace elsa::mr;

struct TestStatsCounter {
    bool intest = false;
    MemoryResource resource;

    /* for complex type */
    std::set<const void*> preTestConstructed;
    std::set<const void*> constructed;
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
    size_t allocOperations = 0;
};
static TestStatsCounter testStats;

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
    testStats.allocOperations = 0;

    testStats.intest = true;
}
static void VerifyTestStats()
{
    testStats.intest = false;

    CHECK_MESSAGE(testStats.constructed.empty(), "Not all objects were destructed");
    CHECK_MESSAGE(testStats.invalidDestruct == 0, "Uninitialized objects were destructed");
    CHECK_MESSAGE(testStats.doubleConstruct == 0, "Initialized objects were re-initialized");
    CHECK_MESSAGE(testStats.invalidAssign == 0, "Uninitialized objects were assigned");
    CHECK_MESSAGE(testStats.invalidAccess == 0, "Uninitialized objects were accessed");
    CHECK_MESSAGE(testStats.destructCount == 0, "Unexpected destructs occurred");
    CHECK_MESSAGE(testStats.defaultConstruct == 0,
                  "Uninitialized objects were default-constructed");
    CHECK_MESSAGE(testStats.allocations.empty(), "Memory leak detected");
    CHECK_MESSAGE(testStats.invalidFree == 0, "Invalid address was deallocated");

    /* reset all missed allocations */
    for (auto& alloc : testStats.allocations)
        testStats.resource->deallocate(alloc.first, alloc.second.first, alloc.second.second);
}

/* checked resource which keeps track of accesses/allocations */
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

public:
    static MemoryResource make() { return MemoryResource::MakeRef(new CheckedResource()); }
};

/* count how many values in the container match the given requirements */
template <class T>
static size_t CheckAllEqual(const T& t, int val, size_t off)
{
    size_t count = 0;
    for (size_t i = off; i < t.size(); ++i) {
        if (t[i] == val)
            ++count;
    }
    return count;
}
template <class T, class It>
static size_t CheckAllMatch(const T& t, It it, size_t values)
{
    size_t count = 0;
    auto tt = t.begin();
    while (tt != t.end() && values-- > 0) {
        if (*tt == *it)
            ++count;
        ++tt;
        ++it;
    }
    return count;
}

/* rotate bits (to initialize all integer values deterministically with non-null values) */
static unsigned int RotateIntBits(unsigned int i, uint32_t c)
{
    if (c == 0)
        return i;
    return (i >> c) | (i << (sizeof(unsigned int) * 8 - c));
}

/* types to check for proper behavior/verify correct operations
 *  (ValCount = number of ints as payload) => for more than one integer, returns 0 on errors!
 *
 *  Multiple Uninitialized values does not make sense
 */
template <size_t ValCount>
struct ComplexType {
    static_assert(ValCount >= 1, "Payload must be at least 1 integer");

public:
    using tag = type_tags::complex;
    template <class T>
    static constexpr bool is = std::is_same<T, ComplexType>::value;
    static constexpr int initValue = 0;

private:
    int _payload[ValCount] = {0};

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
    void _checkValid(const ComplexType<ValCount>& t, bool access) const
    {
        if (testStats.constructed.count(&t) == 0 && testStats.preTestConstructed.count(&t) == 0)
            ++(access ? testStats.invalidAccess : testStats.invalidAssign);
    }

public:
    ComplexType() { _checkConstruct(); }
    ComplexType(int v)
    {
        for (size_t i = 0; i < ValCount; ++i)
            _payload[i] = RotateIntBits(v, i);
        _checkConstruct();
    }
    ComplexType(const ComplexType<ValCount>& t)
    {
        for (size_t i = 0; i < ValCount; ++i)
            _payload[i] = t._payload[i];
        _checkConstruct();
        _checkValid(t, true);
    }
    ComplexType(ComplexType<ValCount>&& t) noexcept
    {
        for (size_t i = 0; i < ValCount; ++i)
            _payload[i] = t._payload[i];
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
    ComplexType<ValCount>& operator=(const ComplexType<ValCount>& t)
    {
        _checkValid(*this, false);
        if (this == &t)
            return *this;
        _checkValid(t, true);
        for (size_t i = 0; i < ValCount; ++i)
            _payload[i] = t._payload[i];
        return *this;
    }
    ComplexType<ValCount>& operator=(ComplexType<ValCount>&& t)
    {
        _checkValid(*this, false);
        if (this == &t)
            return *this;
        _checkValid(t, true);
        for (size_t i = 0; i < ValCount; ++i)
            _payload[i] = t._payload[i];
        return *this;
    }

    operator int() const
    {
        _checkValid(*this, true);
        for (size_t i = 1; i < ValCount; ++i) {
            if (_payload[i] != static_cast<int>(RotateIntBits(_payload[0], i)))
                return 0;
        }
        return _payload[0];
    }
};
template <size_t ValCount>
struct TrivialType {
    static_assert(ValCount >= 1, "Payload must be at least 1 integer");

public:
    using tag = type_tags::trivial;
    template <class T>
    static constexpr bool is = std::is_same<T, TrivialType>::value;
    static constexpr int initValue = 0;

private:
    int _payload[ValCount] = {0};

public:
    TrivialType() = default;
    TrivialType(int v)
    {
        for (size_t i = 0; i < ValCount; ++i)
            _payload[i] = RotateIntBits(v, i);
    }
    ~TrivialType() { ++testStats.destructCount; }
    TrivialType(const TrivialType<ValCount>&) = default;
    TrivialType(TrivialType<ValCount>&&) = default;
    TrivialType<ValCount>& operator=(const TrivialType<ValCount>&) = default;
    TrivialType<ValCount>& operator=(TrivialType<ValCount>&&) = default;
    operator int() const
    {
        for (size_t i = 1; i < ValCount; ++i) {
            if (_payload[i] != static_cast<int>(RotateIntBits(_payload[0], i)))
                return 0;
        }
        return _payload[0];
    }
};
struct UninitType : public TrivialType<1> {
public:
    using tag = type_tags::uninitialized;
    template <class T>
    static constexpr bool is = std::is_same<T, UninitType>::value;
    static constexpr int initValue = CheckedResource::initValue<int>();

public:
    UninitType() { ++testStats.defaultConstruct; }
    UninitType(int v) : TrivialType<1>(v) {}
    UninitType(const UninitType& t) = default;
    UninitType(UninitType&&) = default;
    UninitType& operator=(const UninitType&) = default;
    UninitType& operator=(UninitType&&) = default;
};

struct Randoms {
private:
    std::random_device _dev;
    std::mt19937 _engine;
    std::uniform_int_distribution<size_t> _count;
    std::uniform_int_distribution<int> _value;

private:
    Randoms() : _engine(_dev())
    {
        _count = std::uniform_int_distribution<size_t>(4, 80);
        _value = std::uniform_int_distribution<int>(-100000000, 100000000);
    }

private:
    size_t makeCount() { return _count(_engine); }
    int makeValue() { return _value(_engine); }

public:
    static size_t count() { return Randoms().makeCount(); }
    static int value() { return Randoms().makeValue(); }
    template <class Type>
    static std::list<Type> typeList()
    {
        Randoms _r;
        std::list<Type> out;
        size_t tmp = _r.makeCount();

        for (size_t i = 0; i < tmp; ++i)
            out.push_back(_r.makeValue());
        return out;
    }
    template <class Type>
    static std::vector<Type> typeVec()
    {
        Randoms _r;
        std::vector<Type> out;
        size_t tmp = _r.makeCount();

        for (size_t i = 0; i < tmp; ++i)
            out.push_back(_r.makeValue());
        return out;
    }
};

TEST_SUITE_BEGIN("memoryresources");
TYPE_TO_STRING(ComplexType<1>);
TYPE_TO_STRING(TrivialType<1>);
TYPE_TO_STRING(UninitType);
TYPE_TO_STRING(ComplexType<6>);
TYPE_TO_STRING(TrivialType<8>);

TEST_CASE_TEMPLATE("ContiguousVector::Constructors", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    SUBCASE("Default constructor")
    {
        StartTestStats();

        {
            Vector storage(mres);
            CHECK(storage.size() == 0);
        }

        VerifyTestStats();
    }

    int intCount0 = 50;
    SUBCASE("Constructor(size_t)")
    {
        StartTestStats();

        {
            Vector storage(intCount0, mres);
            CHECK(storage.size() == intCount0);
            CHECK(CheckAllEqual(storage, T::initValue, 0) == intCount0);
        }

        VerifyTestStats();
    }

    int intNum1 = 32, intCount1 = 17;
    SUBCASE("Constructor(size_t, const value&)")
    {
        T temp(intNum1);

        StartTestStats();

        {
            Vector storage(intCount1, temp, mres);
            CHECK(storage.size() == intCount1);
            CHECK(CheckAllEqual(storage, intNum1, 0) == intCount1);
        }

        VerifyTestStats();
    }

    std::vector<int> intVec0 = {5, 9, 123, 130, -123, 2394, 591, 1203, 523, 123};
    SUBCASE("Constructor(ItType, ItType) [consecutive]")
    {
        StartTestStats();

        {
            Vector storage(intVec0.begin(), intVec0.end(), mres);
            CHECK(storage.size() == intVec0.size());
            CHECK(CheckAllMatch(storage, intVec0.begin(), intVec0.size()) == intVec0.size());
        }

        VerifyTestStats();
    }

    std::list<int> intList = {6, 123, -123, 504, 12321, 889123, 345, -543, -8123, 10148};
    SUBCASE("Constructor(ItType, ItType) [not-consecutive]")
    {
        StartTestStats();

        {
            Vector storage(intList.begin(), intList.end(), mres);
            CHECK(storage.size() == intList.size());
            CHECK(CheckAllMatch(storage, intList.begin(), intList.size()) == intList.size());
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(initializer_list) [implicit: consecutive]")
    {
        std::initializer_list<T> intInit = {1, -12, 0, -1123, 0532, 7123, -1239, -67345, 012, -4};

        StartTestStats();

        {
            Vector storage(intInit, mres);
            CHECK(storage.size() == intInit.size());
            CHECK(CheckAllMatch(storage, intInit.begin(), intInit.size()) == intInit.size());
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(const self_type&) [implicit: consecutive]")
    {
        Vector intSelf({-1, 0, 123, 5, -123, 9348, -239411, -123, 64223, -123}, mres);

        StartTestStats();

        {
            Vector storage(intSelf, mres);
            CHECK(storage.size() == intSelf.size());
            CHECK(CheckAllMatch(storage, intSelf.begin(), intSelf.size()) == intSelf.size());
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(self_type&&) [implicit: consecutive]")
    {
        Vector intSelf(intList.begin(), intList.end(), mres);

        StartTestStats();

        {
            Vector storage(std::move(intSelf));
            // NOLINTNEXTLINE(*-use-after-move)
            CHECK(intSelf.size() == 0);
            CHECK(storage.size() == intList.size());
            CHECK(CheckAllMatch(storage, intList.begin(), intList.size()) == intList.size());

            /* necessary as move-constructor 'steals' the values from the pre-constructed storage */
            std::swap(testStats.preTestConstructed, testStats.constructed);
            std::swap(testStats.preTestAllocated, testStats.allocations);
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_CASE_TEMPLATE("ContiguousVector::operator=", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    std::vector<int> intVec0 = {-6454, -4728, -1351, 4778, -283, 356, 6839};
    std::vector<int> intVec1 = {142, -165, 6561, 315, -3790, 313, 988872, 132109, -1970, 12319};
    SUBCASE("operator=(const self_type&)")
    {
        Vector intSelf(intVec1.begin(), intVec1.end(), mres);

        StartTestStats();

        {
            Vector storage(intVec0.begin(), intVec0.end(), mres);

            storage = intSelf;

            CHECK(storage.size() == intVec1.size());
            CHECK(intSelf.size() == intVec1.size());
            CHECK(CheckAllMatch(storage, intVec1.begin(), intVec1.size()) == intVec1.size());
            CHECK(CheckAllMatch(intSelf, intVec1.begin(), intVec1.size()) == intVec1.size());
        }

        VerifyTestStats();
    }

    std::vector<int> intVec2 = {42128, -1117, 703, 2051, 3929, 8967, 5836, -4143, 313842, 230};
    SUBCASE("operator=(self_type&&)")
    {
        Vector intSelf(intVec2.begin(), intVec2.end(), mres);

        StartTestStats();

        {
            Vector storage(intVec0.begin(), intVec0.end(), mres);

            storage = std::move(intSelf);

            CHECK(storage.size() == intVec2.size());
            // NOLINTNEXTLINE(*-use-after-move)
            CHECK(intSelf.size() == 0);
            CHECK(CheckAllMatch(storage, intVec2.begin(), intVec2.size()) == intVec2.size());

            /* necessary as move-constructor 'steals' the values from the pre-constructed storage */
            std::swap(testStats.preTestConstructed, testStats.constructed);
            std::swap(testStats.preTestAllocated, testStats.allocations);
        }

        VerifyTestStats();
    }

    SUBCASE("operator=(initializer_list)")
    {
        std::initializer_list<T> intInit = {-12011, 7768,  5991, 4419,  -5819, -7972,
                                            -1214,  -8911, 9401, -2669, -4947, -2232};

        StartTestStats();

        {
            Vector storage(intVec0.begin(), intVec0.end(), mres);

            storage = intInit;

            CHECK(storage.size() == intInit.size());
            CHECK(CheckAllMatch(storage, intInit.begin(), intInit.size()) == intInit.size());
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_CASE_TEMPLATE("ContiguousVector::assign", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    SUBCASE("assign_default(size_t)")
    {
        auto vec = Randoms::typeVec<T>();
        auto count = Randoms::count();

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign_default(count);

            CHECK(storage.size() == count);

            /* if new size is smaller than vector, uninitialized will not re-initialize */
            if (UninitType::is<T> && count < vec.size())
                CHECK(CheckAllMatch(storage, vec.begin(), count) == count);
            else
                CHECK(CheckAllEqual(storage, T::initValue, 0) == count);
        }

        VerifyTestStats();
    }

    SUBCASE("assign(size_t, const value&)")
    {
        auto vec = Randoms::typeVec<T>();
        auto value = Randoms::value();
        auto count = Randoms::count();

        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign(count, temp);

            CHECK(storage.size() == count);
            CHECK(CheckAllEqual(storage, value, 0) == count);
        }

        VerifyTestStats();
    }

    SUBCASE("assign(ItType, ItType) [consecutive]")
    {
        auto vec0 = Randoms::typeVec<T>();
        auto vec1 = Randoms::typeVec<T>();

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage.assign(vec1.begin(), vec1.end());

            CHECK(storage.size() == vec1.size());
            CHECK(CheckAllMatch(storage, vec1.begin(), vec1.size()) == vec1.size());
        }

        VerifyTestStats();
    }

    SUBCASE("assign(ItType, ItType) [non-consecutive]")
    {
        auto vec = Randoms::typeVec<T>();
        auto list = Randoms::typeList<T>();

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign(list.begin(), list.end());

            CHECK(storage.size() == list.size());
            CHECK(CheckAllMatch(storage, list.begin(), list.size()) == list.size());
        }

        VerifyTestStats();
    }

    SUBCASE("assign(initializer_list) [implicit: consecutive]")
    {
        auto vec = Randoms::typeVec<T>();
        std::initializer_list<T> init = {142, -165,   6561,   315,   -3790,
                                         313, 988872, 132109, -1970, 12319};

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign(init);

            CHECK(storage.size() == init.size());
            CHECK(CheckAllMatch(storage, init.begin(), init.size()) == init.size());
        }

        VerifyTestStats();
    }

    SUBCASE("assign(const self_type&)")
    {
        auto vec0 = Randoms::typeVec<T>();
        auto vec1 = Randoms::typeVec<T>();
        Vector self(vec1.begin(), vec1.end(), mres);

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage.assign(self);

            CHECK(storage.size() == vec1.size());
            CHECK(self.size() == vec1.size());
            CHECK(CheckAllMatch(storage, vec1.begin(), vec1.size()) == vec1.size());
            CHECK(CheckAllMatch(self, vec1.begin(), vec1.size()) == vec1.size());
        }

        VerifyTestStats();
    }
}

TEST_SUITE_END();

/* test that ptr is != to zero on allocations and else zero */
/* test first capacity and grow of it */
/* test when resource is changed */

/* add test for 'uninitialization' */
/* add test for 'trivial' */
/* add test for 'exception-handling' */