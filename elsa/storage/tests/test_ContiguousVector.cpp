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

/* check if all values match the given requirements */
template <class FIt>
static bool CheckAllEqual(FIt f, int val, size_t values)
{
    while (values-- > 0) {
        if (*f != val)
            return false;
        ++f;
    }
    return true;
}
template <class FIt, class SIt>
static bool CheckAllMatch(FIt f, SIt s, size_t values)
{
    while (values-- > 0) {
        if (*f != *s)
            return false;
        ++f;
        ++s;
    }
    return true;
}

/* generate random counts/values or lists/vectors */
struct Randoms {
private:
    std::random_device _dev;
    std::mt19937 _engine;
    std::uniform_int_distribution<size_t> _count;
    std::uniform_int_distribution<int> _value;

private:
    Randoms(size_t min = 4, size_t max = 80) : _engine(_dev())
    {
        _count = std::uniform_int_distribution<size_t>(min, max);
        _value = std::uniform_int_distribution<int>(-100000000, 100000000);
    }

private:
    size_t makeCount() { return _count(_engine); }
    int makeValue() { return _value(_engine); }

public:
    static size_t count(bool emptyable)
    {
        if (emptyable)
            return Randoms(0).makeCount();
        return Randoms().makeCount();
    }
    static size_t index(size_t last) { return Randoms(0, last).makeCount(); }
    static int value() { return Randoms().makeValue(); }
    template <class Type>
    static std::list<Type> list(bool emptyable)
    {
        Randoms _r = emptyable ? Randoms(0) : Randoms();
        std::list<Type> out;
        size_t tmp = _r.makeCount();

        for (size_t i = 0; i < tmp; ++i)
            out.push_back(_r.makeValue());
        return out;
    }
    template <class Type>
    static std::vector<Type> vec(bool emptyable)
    {
        Randoms _r = emptyable ? Randoms(0) : Randoms();
        std::vector<Type> out;
        size_t tmp = _r.makeCount();

        for (size_t i = 0; i < tmp; ++i)
            out.push_back(_r.makeValue());
        return out;
    }
};

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

    SUBCASE("Constructor(size_t)")
    {
        auto count = Randoms::count(true);

        StartTestStats();

        {
            Vector storage(count, mres);

            CHECK(storage.size() == count);
            CHECK(CheckAllEqual(storage.begin(), T::initValue, count));
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(size_t, const value&)")
    {
        auto value = Randoms::value();
        auto count = Randoms::count(true);
        T temp(value);

        StartTestStats();

        {
            Vector storage(count, temp, mres);

            CHECK(storage.size() == count);
            CHECK(CheckAllEqual(storage.begin(), value, count));
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(ItType, ItType) [consecutive]")
    {
        auto vec = Randoms::vec<T>(true);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            CHECK(storage.size() == vec.size());
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(ItType, ItType) [not-consecutive]")
    {
        auto list = Randoms::list<T>(true);

        StartTestStats();

        {
            Vector storage(list.begin(), list.end(), mres);

            CHECK(storage.size() == list.size());
            CHECK(CheckAllMatch(storage.begin(), list.begin(), list.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(initializer_list)")
    {
        std::initializer_list<T> init = {-398209, 479982,  235255,  709892, 967199,
                                         697198,  818550,  1245640, 85076,  -440662,
                                         1493450, 1491831, 379199,  544652, 935449};

        StartTestStats();

        {
            Vector storage(init, mres);

            CHECK(storage.size() == init.size());
            CHECK(CheckAllMatch(storage.begin(), init.begin(), init.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(const self_type&)")
    {
        auto vec = Randoms::vec<T>(true);
        Vector self(vec.begin(), vec.end(), mres);

        StartTestStats();

        {
            Vector storage(self, mres);

            CHECK(storage.size() == vec.size());
            CHECK(self.size() == vec.size());
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));
            CHECK(CheckAllMatch(self.begin(), vec.begin(), vec.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("Constructor(self_type&&)")
    {
        auto vec = Randoms::vec<int>(true);
        Vector self(vec.begin(), vec.end(), mres);

        StartTestStats();

        {
            Vector storage(std::move(self));

            // NOLINTNEXTLINE(*-use-after-move)
            CHECK(self.size() == 0);
            CHECK(storage.size() == vec.size());
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));

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

    SUBCASE("operator=(const self_type&)")
    {
        auto vec0 = Randoms::vec<T>(true);
        auto vec1 = Randoms::vec<T>(true);

        Vector self(vec1.begin(), vec1.end(), mres);

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage = self;

            CHECK(storage.size() == vec1.size());
            CHECK(self.size() == vec1.size());
            CHECK(CheckAllMatch(storage.begin(), vec1.begin(), vec1.size()));
            CHECK(CheckAllMatch(self.begin(), vec1.begin(), vec1.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("operator=(self_type&&)")
    {
        auto vec0 = Randoms::vec<int>(true);
        auto vec1 = Randoms::vec<int>(true);
        Vector self(vec1.begin(), vec1.end(), mres);

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage = std::move(self);

            CHECK(storage.size() == vec1.size());
            // NOLINTNEXTLINE(*-use-after-move)
            CHECK(self.size() == 0);
            CHECK(CheckAllMatch(storage.begin(), vec1.begin(), vec1.size()));

            /* necessary as move-constructor 'steals' the values from the pre-constructed storage */
            std::swap(testStats.preTestConstructed, testStats.constructed);
            std::swap(testStats.preTestAllocated, testStats.allocations);
        }

        VerifyTestStats();
    }

    SUBCASE("operator=(initializer_list)")
    {
        auto vec = Randoms::vec<T>(true);
        std::initializer_list<T> init = {1208778, 857116,  -128880, 1400514, 718035,
                                         1311674, 657510,  1202315, 1264138, -329564,
                                         -191651, -231824, 100457,  -403368, 1025825};

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage = init;

            CHECK(storage.size() == init.size());
            CHECK(CheckAllMatch(storage.begin(), init.begin(), init.size()));
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
        auto vec = Randoms::vec<T>(true);
        auto count = Randoms::count(true);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign_default(count);

            CHECK(storage.size() == count);

            /* if new size is smaller/equal than vector, uninitialized will not re-initialize */
            if (UninitType::is<T> && count <= vec.size())
                CHECK(CheckAllMatch(storage.begin(), vec.begin(), count));
            else
                CHECK(CheckAllEqual(storage.begin(), T::initValue, count));
        }

        VerifyTestStats();
    }

    SUBCASE("assign(size_t, const value&)")
    {
        auto vec = Randoms::vec<T>(true);
        auto value = Randoms::value();
        auto count = Randoms::count(true);
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign(count, temp);

            CHECK(storage.size() == count);
            CHECK(CheckAllEqual(storage.begin(), value, count));
        }

        VerifyTestStats();
    }

    SUBCASE("assign(ItType, ItType) [consecutive]")
    {
        auto vec0 = Randoms::vec<T>(true);
        auto vec1 = Randoms::vec<T>(true);

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage.assign(vec1.begin(), vec1.end());

            CHECK(storage.size() == vec1.size());
            CHECK(CheckAllMatch(storage.begin(), vec1.begin(), vec1.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("assign(ItType, ItType) [non-consecutive]")
    {
        auto vec = Randoms::vec<T>(true);
        auto list = Randoms::list<T>(true);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign(list.begin(), list.end());

            CHECK(storage.size() == list.size());
            CHECK(CheckAllMatch(storage.begin(), list.begin(), list.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("assign(initializer_list)")
    {
        auto vec = Randoms::vec<T>(true);
        std::initializer_list<T> init = {120003,  -343400, 375390,  1224936, 8455,
                                         -312424, 1456582, 768649,  1393532, 586647,
                                         1093424, 1071102, -205884, 118497,  -326290};

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.assign(init);

            CHECK(storage.size() == init.size());
            CHECK(CheckAllMatch(storage.begin(), init.begin(), init.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("assign(const self_type&)")
    {
        auto vec0 = Randoms::vec<T>(true);
        auto vec1 = Randoms::vec<T>(true);
        Vector self(vec1.begin(), vec1.end(), mres);

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage.assign(self);

            CHECK(storage.size() == vec1.size());
            CHECK(self.size() == vec1.size());
            CHECK(CheckAllMatch(storage.begin(), vec1.begin(), vec1.size()));
            CHECK(CheckAllMatch(self.begin(), vec1.begin(), vec1.size()));
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_CASE_TEMPLATE("ContiguousVector::insert", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    SUBCASE("insert_default(ItType, size_t)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size()), count = Randoms::count(true);
        size_t pre = index, post = vec.size() - index;

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.insert_default(storage.begin() + index, count);

            CHECK(storage.size() == vec.size() + count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + count), vec.begin() + pre, post));
            CHECK(CheckAllEqual(storage.begin() + index, T::initValue, count));
        }

        VerifyTestStats();
    }

    SUBCASE("insert(ItType, const value&)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size()), count = 1;
        size_t pre = index, post = vec.size() - index;
        auto value = Randoms::value();
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.insert(storage.begin() + index, temp);

            CHECK(storage.size() == vec.size() + count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + count), vec.begin() + pre, post));
            CHECK(CheckAllEqual(storage.begin() + index, value, count));
        }

        VerifyTestStats();
    }

    SUBCASE("insert(ItType, value&&)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size()), count = 1;
        size_t pre = index, post = vec.size() - index;
        auto value = Randoms::value();
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.insert(storage.begin() + index, std::move(temp));

            CHECK(storage.size() == vec.size() + count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + count), vec.begin() + pre, post));
            CHECK(CheckAllEqual(storage.begin() + index, value, count));
        }

        VerifyTestStats();
    }

    SUBCASE("insert(ItType, size_t, const value&)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size()), count = Randoms::count(true);
        size_t pre = index, post = vec.size() - index;
        auto value = Randoms::value();
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.insert(storage.begin() + index, count, temp);

            CHECK(storage.size() == vec.size() + count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + count), vec.begin() + pre, post));
            CHECK(CheckAllEqual(storage.begin() + index, value, count));
        }

        VerifyTestStats();
    }

    SUBCASE("insert(ItType, ItType, ItType) [consecutive]")
    {
        auto vec0 = Randoms::vec<T>(true);
        auto vec1 = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec0.size());
        size_t pre = index, post = vec0.size() - index;

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            storage.insert(storage.begin() + index, vec1.begin(), vec1.end());

            CHECK(storage.size() == vec0.size() + vec1.size());
            CHECK(CheckAllMatch(storage.begin(), vec0.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + vec1.size()), vec0.begin() + pre, post));
            CHECK(CheckAllMatch(storage.begin() + index, vec1.begin(), vec1.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("insert(ItType, ItType, ItType) [non-consecutive]")
    {
        auto vec = Randoms::vec<T>(true);
        auto list = Randoms::list<T>(true);
        size_t index = Randoms::index(vec.size());
        size_t pre = index, post = vec.size() - index;

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.insert(storage.begin() + index, list.begin(), list.end());

            CHECK(storage.size() == vec.size() + list.size());
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + list.size()), vec.begin() + pre, post));
            CHECK(CheckAllMatch(storage.begin() + index, list.begin(), list.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("insert(ItType, initializer_list)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size());
        size_t pre = index, post = vec.size() - index;
        std::initializer_list<T> init = {-490882, 4245,    -497502, 1038916, 861215,
                                         -6817,   1461747, 332086,  -480681, -362049,
                                         -250110, -371747, -228368, -431462, -292826};

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.insert(storage.begin() + index, init);

            CHECK(storage.size() == vec.size() + init.size());
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + init.size()), vec.begin() + pre, post));
            CHECK(CheckAllMatch(storage.begin() + index, init.begin(), init.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("emplace(ItType, Args&&...)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size()), count = 1;
        size_t pre = index, post = vec.size() - index;
        auto value = Randoms::value();

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.emplace(storage.begin() + index, value);

            CHECK(storage.size() == vec.size() + count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + (index + count), vec.begin() + pre, post));
            CHECK(CheckAllEqual(storage.begin() + index, value, count));
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_CASE_TEMPLATE("ContiguousVector::erase", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    SUBCASE("erase(ItType)")
    {
        auto vec = Randoms::vec<T>(false);
        size_t count = 1;
        size_t index = Randoms::index(vec.size() - count);
        size_t pre = index, post = vec.size() - index - count;

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.erase(storage.begin() + index);

            CHECK(storage.size() == vec.size() - count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + index, vec.begin() + (pre + count), post));
        }

        VerifyTestStats();
    }

    SUBCASE("erase(const ItType)")
    {
        auto vec = Randoms::vec<T>(false);
        size_t count = 1;
        size_t index = Randoms::index(vec.size() - count);
        size_t pre = index, post = vec.size() - index - count;

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.erase(storage.cbegin() + index);

            CHECK(storage.size() == vec.size() - count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + index, vec.begin() + (pre + count), post));
        }

        VerifyTestStats();
    }

    SUBCASE("erase(ItType, ItType)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size());
        size_t count = Randoms::index(vec.size() - index);
        size_t pre = index, post = vec.size() - index - count;

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.erase(storage.begin() + index, storage.begin() + (index + count));

            CHECK(storage.size() == vec.size() - count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + index, vec.begin() + (pre + count), post));
        }

        VerifyTestStats();
    }

    SUBCASE("erase(const ItType, const ItType)")
    {
        auto vec = Randoms::vec<T>(true);
        size_t index = Randoms::index(vec.size());
        size_t count = Randoms::index(vec.size() - index);
        size_t pre = index, post = vec.size() - index - count;

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.erase(storage.cbegin() + index, storage.cbegin() + (index + count));

            CHECK(storage.size() == vec.size() - count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllMatch(storage.begin() + index, vec.begin() + (pre + count), post));
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_CASE_TEMPLATE("ContiguousVector:: modifier", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    SUBCASE("clear()")
    {
        auto vec = Randoms::vec<T>(true);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.clear();

            CHECK(storage.size() == 0);
        }

        VerifyTestStats();
    }

    SUBCASE("emplace_back(Args&&...)")
    {
        auto vec = Randoms::vec<T>(true);
        auto value = Randoms::value();

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.emplace_back(value);

            CHECK(storage.size() == vec.size() + 1);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));
            CHECK(*(storage.begin() + vec.size()) == value);
        }

        VerifyTestStats();
    }

    SUBCASE("push_back(const value&)")
    {
        auto vec = Randoms::vec<T>(true);
        auto value = Randoms::value();
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.push_back(temp);

            CHECK(storage.size() == vec.size() + 1);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));
            CHECK(*(storage.begin() + vec.size()) == value);
        }

        VerifyTestStats();
    }

    SUBCASE("push_back(value&&)")
    {
        auto vec = Randoms::vec<T>(true);
        auto value = Randoms::value();
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.push_back(std::move(temp));

            CHECK(storage.size() == vec.size() + 1);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));
            CHECK(*(storage.begin() + vec.size()) == value);
        }

        VerifyTestStats();
    }

    SUBCASE("pop_back()")
    {
        auto vec = Randoms::vec<T>(false);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.pop_back();

            CHECK(storage.size() == vec.size() - 1);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size() - 1));
        }

        VerifyTestStats();
    }

    SUBCASE("resize(count)")
    {
        auto vec = Randoms::vec<T>(true);
        auto count = Randoms::count(true);
        size_t pre = std::min(count, vec.size());

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.resize(count);

            CHECK(storage.size() == count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllEqual(storage.begin() + pre, T::initValue, count - pre));
        }

        VerifyTestStats();
    }

    SUBCASE("resize(count, const value&)")
    {
        auto vec = Randoms::vec<T>(true);
        auto count = Randoms::count(true);
        size_t pre = std::min(count, vec.size());
        auto value = Randoms::value();
        T temp(value);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.resize(count, temp);

            CHECK(storage.size() == count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), pre));
            CHECK(CheckAllEqual(storage.begin() + pre, value, count - pre));
        }

        VerifyTestStats();
    }

    SUBCASE("swap(self_type&)")
    {
        auto vec0 = Randoms::vec<T>(true);
        auto vec1 = Randoms::vec<T>(true);

        StartTestStats();

        {
            Vector storage0(vec0.begin(), vec0.end(), mres);
            Vector storage1(vec1.begin(), vec1.end(), mres);

            storage0.swap(storage1);

            CHECK(storage0.size() == vec1.size());
            CHECK(storage1.size() == vec0.size());
            CHECK(CheckAllMatch(storage0.begin(), vec1.begin(), vec1.size()));
            CHECK(CheckAllMatch(storage1.begin(), vec0.begin(), vec0.size()));
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_CASE_TEMPLATE("ContiguousVector:: capacity", T, ComplexType<1>, TrivialType<1>, UninitType,
                   ComplexType<6>, TrivialType<8>)
{
    MemoryResource mres = CheckedResource::make();
    testStats.resource = mres;
    using Vector = ContiguousVector<T, typename T::tag, detail::ContPointer, detail::ContIterator>;

    SUBCASE("size()")
    {
        auto vec = Randoms::vec<T>(false);

        StartTestStats();

        {
            Vector storage(mres);

            for (size_t i = 0; i < vec.size(); ++i) {
                CHECK(storage.size() == i);
                storage.push_back(vec[i]);
            }
            CHECK(storage.size() == vec.size());
        }

        VerifyTestStats();
    }

    SUBCASE("empty()")
    {
        auto vec0 = Randoms::vec<T>(false);
        auto vec1 = Randoms::vec<T>(false);
        auto vec2 = Randoms::vec<T>(false);

        StartTestStats();

        {
            Vector storage(vec0.begin(), vec0.end(), mres);

            CHECK(!storage.empty());
            storage.clear();
            CHECK(storage.empty());

            storage.assign(vec1.begin(), vec1.end());

            CHECK(!storage.empty());
            storage.clear();
            CHECK(storage.empty());

            storage.assign(vec2.begin(), vec2.end());

            CHECK(!storage.empty());
            storage.clear();
            CHECK(storage.empty());
        }

        VerifyTestStats();
    }

    SUBCASE("max_size()")
    {
        auto vec = Randoms::vec<T>(false);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            CHECK(storage.max_size() == std::numeric_limits<ptrdiff_t>::max());
        }

        VerifyTestStats();
    }

    SUBCASE("reserve()")
    {
        auto vec = Randoms::vec<T>(true);
        auto count = Randoms::count(true);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            storage.reserve(count);

            CHECK(storage.size() == vec.size());
            CHECK(storage.capacity() >= count);
            CHECK(CheckAllMatch(storage.begin(), vec.begin(), vec.size()));
        }

        VerifyTestStats();
    }

    SUBCASE("capacity()")
    {
        auto vec0 = Randoms::vec<T>(false);
        auto vec1 = Randoms::vec<T>(false);

        StartTestStats();

        {
            Vector storage(mres);

            CHECK(storage.capacity() >= storage.size());

            /* append between 1 and 4 values and keep track of the
             *  number of insertions and capacity-changes */
            size_t last = storage.capacity(), changes = 0, touched = 0;
            while (storage.size() < vec0.size()) {
                size_t count = std::min(1 + Randoms::index(3), vec0.size() - storage.size());
                auto it = vec0.begin() + storage.size();

                storage.insert(storage.end(), it, it + count);

                CHECK(storage.capacity() >= storage.size());

                ++touched;
                if (storage.capacity() == last)
                    continue;
                ++changes;
                last = storage.capacity();
            }

            /* check that the capacity-adjustments preserved the content */
            CHECK(CheckAllMatch(storage.begin(), vec0.begin(), vec0.size()));

            /* rudimentary check to ensure that not every resize triggers a capacity-change */
            CHECK(changes <= touched);
            if (touched > 8)
                CHECK(changes < touched);

            storage.insert(storage.end(), vec1.begin(), vec1.end());
            CHECK(storage.capacity() >= storage.size());

            /* check that size-reductions do not lead to capacity-changes */
            size_t cap = storage.capacity();
            while (!storage.empty()) {
                size_t count = std::min(Randoms::index(4), storage.size());

                storage.resize(storage.size() - count);
                CHECK(storage.capacity() == cap);
            }

            storage.clear();
            CHECK(storage.capacity() == cap);
        }

        VerifyTestStats();
    }

    SUBCASE("shrink_to_fit()")
    {
        auto vec = Randoms::vec<T>(false);

        StartTestStats();

        {
            Vector storage(vec.begin(), vec.end(), mres);

            while (!storage.empty()) {
                size_t count = std::min(1 + Randoms::index(3), storage.size());

                size_t cap = storage.capacity();
                storage.resize(storage.size() - count);
                CHECK(storage.capacity() == cap);

                storage.shrink_to_fit();
                CHECK(storage.capacity() == storage.size());

                /* check that the capacity-adjustments preserved the content */
                CHECK(CheckAllMatch(storage.begin(), vec.begin(), storage.size()));
            }

            storage.shrink_to_fit();
            CHECK(storage.capacity() == 0);
        }

        VerifyTestStats();
    }

    testStats.resource.release();
}

TEST_SUITE_END();

/* test on empty: begin() == end() */
/* test that ptr is != to zero on allocations and else zero */
/* test when resource is changed */
/* add test for 'exception-handling' */