#pragma once

#ifndef TILEXR_CHECKER_SHIM_KERNEL_OPERATOR_H_
#define TILEXR_CHECKER_SHIM_KERNEL_OPERATOR_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef TILEXR_CHECKER_ENABLE_PIPE_TRACE_HOOK
#include "tilexr/checker/trace_runtime.h"
#endif

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef FORCE_INLINE_AICORE
#define FORCE_INLINE_AICORE inline
#endif

namespace AscendC {

namespace checker_source {
inline const char *File(const char *file = __builtin_FILE()) { return file; }
inline int Line(int line = __builtin_LINE()) { return line; }
}  // namespace checker_source

#ifndef TILEXR_CHECKER_CUSTOM_BLOCK_HOOK
namespace checker_hook {
inline int GetBlockIdx() { return 0; }
inline int GetBlockNum() { return 1; }
}  // namespace checker_hook
#endif

#ifndef TILEXR_CHECKER_ENABLE_PIPE_TRACE_HOOK
namespace checker_pipe_hook {
inline void RecordPipeSet(int, int, const char *, int) {}
inline void RecordPipeWait(int, int, const char *, int) {}
inline void RecordPipeBarrier(int, const char *, int) {}
}  // namespace checker_pipe_hook
#else
namespace checker_pipe_hook {
inline tilexr::checker::SourceLocation PipeLoc(const char *file, int line)
{
    return tilexr::checker::SourceLocation{file, line};
}

inline void RecordPipeSet(int pipe, int event_id, const char *file, int line)
{
    auto *runtime = tilexr::checker::TraceRuntime::Current();
    if (runtime != nullptr) {
        runtime->RecordPipeEvent(tilexr::checker::EventKind::kPipeSet, pipe, event_id,
                                 PipeLoc(file, line), "trace AscendC::SetFlag");
    }
}

inline void RecordPipeWait(int pipe, int event_id, const char *file, int line)
{
    auto *runtime = tilexr::checker::TraceRuntime::Current();
    if (runtime != nullptr) {
        runtime->RecordPipeEvent(tilexr::checker::EventKind::kPipeWait, pipe, event_id,
                                 PipeLoc(file, line), "trace AscendC::WaitFlag");
    }
}

inline void RecordPipeBarrier(int pipe, const char *file, int line)
{
    auto *runtime = tilexr::checker::TraceRuntime::Current();
    if (runtime != nullptr) {
        runtime->RecordPipeEvent(tilexr::checker::EventKind::kPipeBarrier, pipe, -1,
                                 PipeLoc(file, line), "trace AscendC::PipeBarrier");
    }
}
}  // namespace checker_pipe_hook
#endif

using event_t = int;
using TEventID = int;

enum class HardEvent {
    MTE2_MTE3,
    MTE2_S,
    MTE2_V,
    MTE3_MTE2,
    MTE3_S,
    S_MTE2,
    S_MTE3,
    S_V,
    V_MTE2,
    V_MTE3,
    V_S
};

enum class QuePosition { VECCALC };
enum class RoundMode { CAST_NONE };
enum class TPosition { VECIN };

constexpr int EVENT_ID0 = 0;
constexpr int EVENT_ID1 = 1;
constexpr int EVENT_ID2 = 2;
constexpr int EVENT_ID3 = 3;
constexpr int PIPE_ALL = 0;
constexpr int PIPE_V = 1;
constexpr int ALIGN_SIZE = 32;
constexpr uint64_t MASK_PLACEHOLDER = 0;

template <HardEvent event>
constexpr int PipeForEvent()
{
    return event == HardEvent::MTE2_V || event == HardEvent::S_V ||
                   event == HardEvent::V_MTE2 ||
                   event == HardEvent::V_MTE3 || event == HardEvent::V_S
               ? PIPE_V
               : PIPE_ALL;
}

struct TBuffAddr {
    uint64_t bufferAddr = 0;
    uint8_t logicPos = 0;
};

struct TensorAddress {
    uint8_t logicPos = 0;
    uint64_t bufferAddr = 0;
};

struct DataCopyExtParams {
    DataCopyExtParams() = default;
    DataCopyExtParams(uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                      uint32_t dstStride, uint32_t rsv)
        : blockCount(blockCount), blockLen(blockLen), srcStride(srcStride),
          dstStride(dstStride), rsv(rsv) {}
    uint16_t blockCount = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    uint32_t rsv = 0;
};

template <typename T>
struct DataCopyPadExtParams {
    DataCopyPadExtParams() = default;
    DataCopyPadExtParams(bool isPad, uint8_t leftPadding, uint8_t rightPadding, T paddingValue)
        : isPad(isPad), leftPadding(leftPadding), rightPadding(rightPadding),
          paddingValue(paddingValue) {}
    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue {};
};

template <typename T>
class LocalTensor {
public:
    TensorAddress address_;

    void SetAddr(TBuffAddr addr) { address_.bufferAddr = addr.bufferAddr; }
    LocalTensor<T> operator[](int64_t) const { return LocalTensor<T>(); }
    T GetValue(int64_t) const { return T(); }
    void SetValue(int64_t, T) {}
};

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T *addr, int64_t = 0) { addr_ = addr; }
    T *GetPhyAddr() const { return addr_; }
    GlobalTensor<T> operator[](int64_t offset) const
    {
        GlobalTensor<T> tensor;
        if (addr_ == nullptr) {
            tensor.addr_ = nullptr;
        } else {
            const uintptr_t addr = reinterpret_cast<uintptr_t>(addr_);
            tensor.addr_ = reinterpret_cast<T *>(addr + static_cast<uintptr_t>(offset) * sizeof(T));
        }
        return tensor;
    }
    T GetValue(int64_t) const { return T(); }
    void SetValue(int64_t, T) {}

private:
    T *addr_ = nullptr;
};

template <QuePosition>
class TBuf {
public:
    template <typename T>
    LocalTensor<T> GetWithOffset(int64_t, int64_t)
    {
        return LocalTensor<T>();
    }

    template <typename T>
    void FreeTensor(LocalTensor<T> &) {}
};

class TPipe {
public:
    template <QuePosition position>
    void InitBuffer(TBuf<position> &, int64_t) {}
};

struct GlobalTensorBase {};
struct LocalTensorBase {};

inline int GetBlockIdx() { return checker_hook::GetBlockIdx(); }
inline int GetBlockNum() { return checker_hook::GetBlockNum(); }
inline uint64_t GetSystemCycle() { return 0; }
inline uint64_t get_pc() { return 0; }

template <HardEvent event>
inline void SetFlag(event_t event_id,
                    const char *file = checker_source::File(),
                    int line = checker_source::Line())
{
    checker_pipe_hook::RecordPipeSet(PipeForEvent<event>(), event_id, file, line);
}

template <HardEvent event>
inline void WaitFlag(event_t event_id,
                     const char *file = checker_source::File(),
                     int line = checker_source::Line())
{
    checker_pipe_hook::RecordPipeWait(PipeForEvent<event>(), event_id, file, line);
}

template <int pipe>
inline void PipeBarrier(const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    checker_pipe_hook::RecordPipeBarrier(pipe, file, line);
}

inline void pipe_barrier(int pipe,
                         const char *file = checker_source::File(),
                         int line = checker_source::Line())
{
    checker_pipe_hook::RecordPipeBarrier(pipe, file, line);
}

inline void RecordDataCopy(void *dst, const void *src, size_t bytes,
                           const char *detail, const char *file, int line)
{
#ifdef TILEXR_CHECKER_ENABLE_PIPE_TRACE_HOOK
    auto *runtime = tilexr::checker::TraceRuntime::Current();
    if (runtime != nullptr) {
        runtime->RecordCopy(dst, src, bytes, 0,
                            tilexr::checker::SourceLocation{file, line},
                            detail);
    }
#else
    (void)dst;
    (void)src;
    (void)bytes;
    (void)detail;
    (void)file;
    (void)line;
#endif
}

inline void RecordUnsupportedDataCopy(const char *detail, const char *file, int line)
{
#ifdef TILEXR_CHECKER_ENABLE_PIPE_TRACE_HOOK
    auto *runtime = tilexr::checker::TraceRuntime::Current();
    if (runtime != nullptr) {
        runtime->RecordUnsupportedDataCopy(
            tilexr::checker::SourceLocation{file, line}, detail);
    }
#else
    (void)detail;
    (void)file;
    (void)line;
#endif
}

inline bool IsContiguousNoPad(const DataCopyExtParams &params)
{
    return params.blockCount > 0 && params.blockLen > 0 &&
           params.srcStride == 0 && params.dstStride == 0;
}

template <typename T>
inline bool IsNoPadding(const DataCopyPadExtParams<T> &params)
{
    return !params.isPad && params.leftPadding == 0 && params.rightPadding == 0;
}

template <typename Dst, typename Src>
inline void DataCopy(Dst *dst, const Src *src, size_t bytes,
                     const char *file = checker_source::File(),
                     int line = checker_source::Line())
{
    RecordDataCopy(static_cast<void *>(dst), static_cast<const void *>(src),
                   bytes, "trace AscendC::DataCopy", file, line);
}

template <typename Dst, typename Src>
inline void DataCopy(Dst *dst, Src *src, size_t bytes,
                     const char *file = checker_source::File(),
                     int line = checker_source::Line())
{
    RecordDataCopy(static_cast<void *>(dst), static_cast<const void *>(src),
                   bytes, "trace AscendC::DataCopy", file, line);
}

template <typename... Args>
inline void DataCopy(Args&&...) {}

template <typename Dst, typename Src>
inline void DataCopyPad(Dst *dst, Src *src, const DataCopyExtParams &params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    if (IsContiguousNoPad(params)) {
        RecordDataCopy(static_cast<void *>(dst), static_cast<const void *>(src),
                       static_cast<size_t>(params.blockCount) *
                           static_cast<size_t>(params.blockLen),
                       "trace AscendC::DataCopyPad", file, line);
    } else {
        RecordUnsupportedDataCopy("trace AscendC::DataCopyPad", file, line);
    }
}

template <typename Dst, typename Src>
inline void DataCopyPad(Dst *dst, const Src *src, const DataCopyExtParams &params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    if (IsContiguousNoPad(params)) {
        RecordDataCopy(static_cast<void *>(dst), static_cast<const void *>(src),
                       static_cast<size_t>(params.blockCount) *
                           static_cast<size_t>(params.blockLen),
                       "trace AscendC::DataCopyPad", file, line);
    } else {
        RecordUnsupportedDataCopy("trace AscendC::DataCopyPad", file, line);
    }
}

template <typename Dst, typename Src>
inline void DataCopyPad(Dst *dst, Src *src, DataCopyExtParams &params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    DataCopyPad(dst, src, static_cast<const DataCopyExtParams &>(params),
                file, line);
}

template <typename Dst, typename Src>
inline void DataCopyPad(Dst *dst, const Src *src, DataCopyExtParams &params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    DataCopyPad(dst, src, static_cast<const DataCopyExtParams &>(params),
                file, line);
}

template <typename Dst, typename Src, typename PadT>
inline void DataCopyPad(Dst *dst, Src *src, const DataCopyExtParams &params,
                        const DataCopyPadExtParams<PadT> &pad_params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    if (IsContiguousNoPad(params) && IsNoPadding(pad_params)) {
        RecordDataCopy(static_cast<void *>(dst), static_cast<const void *>(src),
                       static_cast<size_t>(params.blockCount) *
                           static_cast<size_t>(params.blockLen),
                       "trace AscendC::DataCopyPad", file, line);
    } else {
        RecordUnsupportedDataCopy("trace AscendC::DataCopyPad", file, line);
    }
}

template <typename Dst, typename Src, typename PadT>
inline void DataCopyPad(Dst *dst, const Src *src, const DataCopyExtParams &params,
                        const DataCopyPadExtParams<PadT> &pad_params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    if (IsContiguousNoPad(params) && IsNoPadding(pad_params)) {
        RecordDataCopy(static_cast<void *>(dst), static_cast<const void *>(src),
                       static_cast<size_t>(params.blockCount) *
                           static_cast<size_t>(params.blockLen),
                       "trace AscendC::DataCopyPad", file, line);
    } else {
        RecordUnsupportedDataCopy("trace AscendC::DataCopyPad", file, line);
    }
}

template <typename Dst, typename Src, typename PadT>
inline void DataCopyPad(Dst *dst, Src *src, DataCopyExtParams &params,
                        DataCopyPadExtParams<PadT> &pad_params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    DataCopyPad(dst, src, static_cast<const DataCopyExtParams &>(params),
                static_cast<const DataCopyPadExtParams<PadT> &>(pad_params),
                file, line);
}

template <typename Dst, typename Src, typename PadT>
inline void DataCopyPad(Dst *dst, const Src *src, DataCopyExtParams &params,
                        DataCopyPadExtParams<PadT> &pad_params,
                        const char *file = checker_source::File(),
                        int line = checker_source::Line())
{
    DataCopyPad(dst, src, static_cast<const DataCopyExtParams &>(params),
                static_cast<const DataCopyPadExtParams<PadT> &>(pad_params),
                file, line);
}

template <typename... Args>
inline void DataCopyPad(Args&&...) {}

template <typename T>
inline void SetAtomicType() {}

template <typename T>
inline void SetAtomicAdd() {}

template <typename T>
inline void SetAtomicMax() {}

template <typename T>
inline void SetAtomicMin() {}

inline void SetAtomicNone() {}

template <typename... Args>
inline void CastImpl(Args&&...) {}

template <typename... Args>
inline void AddsImpl(Args&&...) {}

template <typename... Args>
inline void MulsImpl(Args&&...) {}

template <typename... Args>
inline void MulImpl(Args&&...) {}

template <typename... Args>
inline void Cast(Args&&...) {}

template <typename... Args>
inline void Muls(Args&&...) {}

template <typename... Args>
inline void Axpy(Args&&...) {}

template <typename... Args>
inline void Div(Args&&...) {}

template <typename... Args>
inline void ReduceMin(Args&&...) {}

template <typename T, bool saturate, typename... Args>
inline void Add(Args&&...) {}

}  // namespace AscendC

using AscendC::EVENT_ID0;
using AscendC::EVENT_ID1;
using AscendC::EVENT_ID2;
using AscendC::EVENT_ID3;
using AscendC::PIPE_ALL;
using AscendC::PIPE_V;
using AscendC::MASK_PLACEHOLDER;
using AscendC::event_t;
using AscendC::TEventID;

#endif  // TILEXR_CHECKER_SHIM_KERNEL_OPERATOR_H_
