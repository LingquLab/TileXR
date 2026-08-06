
#ifndef TILEXR_TEST_KERNEL_OPERATOR_H
#define TILEXR_TEST_KERNEL_OPERATOR_H

#include <cstddef>
#include <cstdint>

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

template <typename T>
inline T ld_dev(T* addr, uint32_t offset)
{
    return addr[offset];
}

template <typename Value, typename T>
inline void st_dev(Value value, T* addr, uint32_t offset)
{
    addr[offset] = static_cast<T>(value);
}

namespace AscendC {

enum class CacheLine {
    SINGLE_CACHE_LINE,
};

enum class DcciDst {
    CACHELINE_OUT,
};

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T* addr)
    {
        addr_ = addr;
    }

    T& operator[](uint64_t offset)
    {
        return addr_[offset];
    }

private:
    T* addr_ = nullptr;
};

template <typename T, CacheLine, DcciDst>
inline void DataCacheCleanAndInvalid(T&)
{
}

} // namespace AscendC

#endif // TILEXR_TEST_KERNEL_OPERATOR_H
