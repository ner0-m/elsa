#include "doctest/doctest.h"

#include "memory_resource/ContiguousStorage.h"
#include "memory_resource/HostStandardResource.h"
#include "Assertions.h"

#include <map>
#include <set>

using namespace elsa::mr;

struct TestStatsCounter {
    /* for complex type */
    std::set<struct ComplexType*> constructed;
    size_t invalidDestruct = 0;
    size_t doubleConstruct = 0;
    size_t invalidAssign = 0;

    /* trivial type */
    size_t unexpectedDestruct = 0;

    /* uninitialized type */
    size_t unexpectedDefConstruct = 0;

    /* for allocator */
    std::map<void*, std::pair<size_t, size_t>> allocations;
    size_t invalidFree = 0;
    size_t memoryOperations = 0;
    size_t allocOperations = 0;

public:
    void reset(MemoryResource& res)
    {
        constructed.clear();
        invalidDestruct = 0;
        doubleConstruct = 0;
        invalidAssign = 0;
        unexpectedDestruct = 0;
        unexpectedDefConstruct = 0;
        invalidFree = 0;
        memoryOperations = 0;
        allocOperations = 0;
        for (auto& alloc : allocations)
            res->deallocate(alloc.first, alloc.second.first, alloc.second.second);
        allocations.clear();
    }
};
static TestStatsCounter testStats;

class CheckedResource : private HostStandardResource
{
private:
    static constexpr uint8_t InitByteValue = 42;

public:
    template <class T>
    static T initValue()
    {
        T t = 0;
        uint8_t* p8 = reinterpret_cast<uint8_t*>(&t);
        for (size_t i = 0; i < sizeof(T); ++i)
            p8[i] = InitByteValue;
        return t;
    }

private:
    void* allocate(size_t size, size_t alignment) override
    {
        void* p = HostStandardResource::allocate(size, alignment);
        testStats.allocations.insert({p, {size, alignment}});
        ++testStats.allocOperations;

        /* fill the content with non-zero values */
        uint8_t* p8 = reinterpret_cast<uint8_t*>(p);
        for (size_t i = 0; i < size; i++)
            p8[i] = InitByteValue;
        return p;
    }
    void deallocate(void* ptr, size_t size, size_t alignment) override
    {
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
private:
    int _payload = 0;

private:
    void _checkConstruct()
    {
        if (testStats.constructed.count(this) > 0)
            ++testStats.doubleConstruct;
        else
            testStats.constructed.insert(this);
    }

public:
    ComplexType() : _payload(0) { _checkConstruct(); }
    ComplexType(int i) : _payload(i) { _checkConstruct(); }
    ComplexType(const ComplexType& t) : _payload(t._payload) { _checkConstruct(); }
    ComplexType(ComplexType&& t) noexcept : _payload(t._payload) { _checkConstruct(); }
    ~ComplexType()
    {
        auto it = testStats.constructed.find(this);
        if (it == testStats.constructed.end())
            ++testStats.invalidDestruct;
        else
            testStats.constructed.erase(it);
    }
    ComplexType& operator=(const ComplexType& t)
    {
        if (testStats.constructed.count(this) == 0) {
            ++testStats.invalidAssign;
            testStats.constructed.insert(this);
        }

        _payload = t._payload;
        return *this;
    }
    ComplexType& operator=(ComplexType&& t)
    {
        if (testStats.constructed.count(this) == 0) {
            ++testStats.invalidAssign;
            testStats.constructed.insert(this);
        }

        _payload = t._payload;
        return *this;
    }

    operator int() const { return _payload; }
};
struct TrivialType {
private:
    int _payload = 0;

public:
    TrivialType() : _payload(0) {}
    TrivialType(int i) : _payload(i) {}
    ~TrivialType() { ++testStats.unexpectedDestruct; }
    TrivialType(const TrivialType&) = default;
    TrivialType(TrivialType&&) = default;
    TrivialType& operator=(const TrivialType&) = default;
    TrivialType& operator=(TrivialType&&) = default;
    operator int() const { return _payload; }
};
struct UninitializedType : public TrivialType {
public:
    UninitializedType() { ++testStats.unexpectedDefConstruct; }
    UninitializedType(int i) : TrivialType(i) {}
    UninitializedType(const UninitializedType&) = default;
    UninitializedType(UninitializedType&&) = default;
    UninitializedType& operator=(const UninitializedType&) = default;
    UninitializedType& operator=(UninitializedType&&) = default;
};

template <class T>
using TestTypeTag =
    std::conditional_t<std::is_same<T, ComplexType>::value, type_tags::complex,
                       std::conditional_t<std::is_same<T, TrivialType>::value, type_tags::trivial,
                                          type_tags::uninitialized>>;
template <class T>
using TestStorage = ContiguousStorage<T, TestTypeTag<T>>;

template <class T>
static constexpr bool only_trivial =
    std::is_base_of_v<type_tags::trivial, TestTypeTag<T>>
    && !std::is_base_of_v<type_tags::uninitialized, TestTypeTag<T>>;
template <class T>
static constexpr bool only_uninitialized =
    std::is_base_of_v<type_tags::uninitialized, TestTypeTag<T>>;

TEST_SUITE_BEGIN("elsa::mr::ContiguousStorage");
TYPE_TO_STRING(UninitializedType);
TYPE_TO_STRING(TrivialType);
TYPE_TO_STRING(ComplexType);

static void VerifyTestStats(size_t destruct, MemoryResource& res)
{
    CHECK_MESSAGE(testStats.constructed.empty(), "Not all objects were destructed");
    CHECK_MESSAGE(testStats.invalidDestruct == 0, "Uninitialized objects were destructed");
    CHECK_MESSAGE(testStats.doubleConstruct == 0, "Initialized objects were re-initialized");
    CHECK_MESSAGE(testStats.invalidAssign == 0, "Uninitialized objects were assigned");
    CHECK_MESSAGE(testStats.unexpectedDestruct == destruct, "Trivial objects were destructed");
    CHECK_MESSAGE(testStats.unexpectedDefConstruct == 0,
                  "Uninitialized objects were default-constructed");
    CHECK_MESSAGE(testStats.allocations.empty(), "Memory leak detected");
    CHECK_MESSAGE(testStats.invalidFree == 0, "Invalid address was deallocated");

    testStats.reset(res);
}

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

TEST_CASE_TEMPLATE("Constructors", T, UninitializedType, TrivialType, ComplexType)
{
    MemoryResource mres = CheckedResource::make();

    SUBCASE("Default constructor")
    {
        {
            TestStorage<T> storage(mres);

            CHECK(storage.resource() == mres);
            CHECK(storage.size() == 0);
            CHECK(storage.capacity() == 0);

            CHECK(testStats.allocOperations == 0);
            CHECK(testStats.memoryOperations == 0);
        }
        VerifyTestStats(0, mres);
    }

    SUBCASE("Constructor(size_t)")
    {
        {
            TestStorage<T> storage(50, mres);

            CHECK(storage.resource() == mres);
            REQUIRE(storage.size() == 50);
            CHECK(storage.capacity() == 50);
            int expected = only_uninitialized<T> ? CheckedResource::initValue<int>() : 0;
            CHECK(CheckAllEqual(storage, expected) == 50);

            CHECK(testStats.allocOperations > 0);
            if (only_trivial<T>)
                CHECK(testStats.memoryOperations > 0);
            else
                CHECK(testStats.memoryOperations == 0);
        }
        VerifyTestStats(only_trivial<T> ? 1 : 0, mres);
    }
}

TEST_SUITE_END();

/* add test for 'uninitialization' */
/* add test for 'trivial' */
/* add test for 'exception-handling' */