
#ifndef TILEXR_TEST_KERNEL_OPERATOR_H
#define TILEXR_TEST_KERNEL_OPERATOR_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
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

enum class HardEvent {
    S_MTE3,
    MTE3_S,
};

using TEventID = uint32_t;

class TPipe {
public:
    TEventID FetchEventID(HardEvent)
    {
        return 0U;
    }
};

template <HardEvent>
inline void SetFlag(TEventID)
{
}

template <HardEvent>
inline void WaitFlag(TEventID)
{
}

struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
    int64_t srcStride;
    int64_t dstStride;
    uint32_t rsv;
};

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

    void SetGlobalBuffer(T* addr, uint64_t size)
    {
        addr_ = addr;
        size_ = size;
    }

    T& operator[](uint64_t offset)
    {
        return addr_[offset];
    }

    T* GetPhyAddr() const
    {
        return addr_;
    }

private:
    T* addr_ = nullptr;
    uint64_t size_ = 0U;
};

template <typename T>
class LocalTensor {
public:
    LocalTensor() = default;

    LocalTensor(T* addr, uint32_t size) : addr_(addr), size_(size)
    {
    }

    uint64_t GetPhyAddr() const
    {
        return reinterpret_cast<uint64_t>(addr_);
    }

    uint32_t GetSize() const
    {
        return size_;
    }

    LocalTensor<T> operator[](uint32_t offset) const
    {
        return LocalTensor<T>(addr_ + offset, size_ - offset);
    }

private:
    T* addr_ = nullptr;
    uint32_t size_ = 0U;
};

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
    const DataCopyExtParams& params)
{
    std::memcpy(dst.GetPhyAddr(), reinterpret_cast<const void*>(src.GetPhyAddr()),
        static_cast<size_t>(params.blockLen) * params.blockCount);
}

template <typename T, CacheLine, DcciDst>
inline void DataCacheCleanAndInvalid(T&)
{
}

} // namespace AscendC

inline AscendC::TPipe* GetTPipePtr()
{
    static AscendC::TPipe pipe;
    return &pipe;
}

#endif // TILEXR_TEST_KERNEL_OPERATOR_H
