/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_H
#define TILEXR_UDMA_H

#include "comm_args.h"
#include "shmem/include/device/gm2gm/engine/shmem_device_udma.h"

/**
 * @file tilexr_udma.h
 * @brief Device-side UDMA wrapper for AICore kernels
 *
 * Usage in AICore kernels:
 *   1. Check if UDMA is enabled: if (TileXR::UDMAEnabled(args)) { ... }
 *   2. Use TileXR::UDMAPutNbi/GetNbi for one-sided transfers
 *   3. Use TileXR::UDMAQuiet() to ensure completion before flag sync
 *   4. Use atomic operations for lock-free coordination
 *
 * All functions are inline and callable from __aicore__ context.
 */

namespace TileXR {

/**
 * @brief Check if UDMA is enabled for this communication
 * @param args CommArgs structure from host
 * @return true if UDMA is initialized and available
 */
__aicore__ inline bool UDMAEnabled(const CommArgs& args) {
    return args.udma_enabled != 0;
}

/**
 * @brief Non-blocking one-sided PUT operation
 * @tparam T Data type (float16, float32, int32, etc.)
 * @param args CommArgs with peer memory pointers
 * @param target_rank Destination rank ID
 * @param local_src Local source address
 * @param remote_offset Offset in remote rank's buffer (in elements of T)
 * @param count Number of elements to transfer
 */
template <typename T>
__aicore__ inline void UDMAPutNbi(const CommArgs& args, int target_rank,
                                   const T* local_src, size_t remote_offset,
                                   size_t count) {
    if (!UDMAEnabled(args)) return;

    auto remote_addr = reinterpret_cast<__gm__ T*>(
        static_cast<char*>(args.peer_mem_ptrs[target_rank]) + remote_offset * sizeof(T));
    aclshmemx_udma_put_nbi(remote_addr, const_cast<__gm__ T*>(local_src),
                           static_cast<__ubuf__ T*>(nullptr), static_cast<uint32_t>(count), target_rank);
}

/**
 * @brief Non-blocking one-sided GET operation
 * @tparam T Data type
 * @param args CommArgs with peer memory pointers
 * @param source_rank Source rank ID
 * @param local_dst Local destination address
 * @param remote_offset Offset in remote rank's buffer (in elements of T)
 * @param count Number of elements to transfer
 */
template <typename T>
__aicore__ inline void UDMAGetNbi(const CommArgs& args, int source_rank,
                                   T* local_dst, size_t remote_offset,
                                   size_t count) {
    if (!UDMAEnabled(args)) return;

    auto remote_addr = reinterpret_cast<__gm__ T*>(
        static_cast<char*>(args.peer_mem_ptrs[source_rank]) + remote_offset * sizeof(T));
    aclshmemx_udma_get_nbi(reinterpret_cast<__gm__ T*>(local_dst), remote_addr,
                           static_cast<__ubuf__ T*>(nullptr), static_cast<uint32_t>(count), source_rank);
}

/**
 * @brief PUT with signal: transfer data then atomically increment remote flag
 * @tparam T Data type
 * @param args CommArgs with peer memory and flag pointers
 * @param target_rank Destination rank ID
 * @param local_src Local source address
 * @param remote_offset Offset in remote rank's buffer (in elements of T)
 * @param count Number of elements to transfer
 * @param flag_offset Offset in remote rank's flag buffer (in uint64_t units)
 */
template <typename T>
__aicore__ inline void UDMAPutSignalNbi(const CommArgs& args, int target_rank,
                                         const T* local_src, size_t remote_offset,
                                         size_t count, size_t flag_offset) {
    if (!UDMAEnabled(args)) return;

    auto remote_addr = reinterpret_cast<__gm__ T*>(
        static_cast<char*>(args.peer_mem_ptrs[target_rank]) + remote_offset * sizeof(T));
    auto remote_flag = reinterpret_cast<__gm__ uint64_t*>(
        static_cast<uint64_t*>(args.peer_flag_ptrs[target_rank]) + flag_offset);

    aclshmemx_udma_put_signal_nbi(remote_addr, const_cast<__gm__ T*>(local_src),
                                  static_cast<uint32_t>(count), remote_flag, 1UL, target_rank);
}

/**
 * @brief Wait for all outstanding UDMA operations to complete
 * @note Must be called before flag-based synchronization to ensure visibility
 */
__aicore__ inline void UDMAQuiet(int target_rank) {
    aclshmemx_udma_quiet(target_rank);
}

/**
 * @brief Atomic add to remote memory (fire-and-forget)
 * @tparam T Integer type (int32_t, uint64_t, etc.)
 * @param args CommArgs with peer flag pointers
 * @param target_rank Destination rank ID
 * @param flag_offset Offset in remote rank's flag buffer (in T units)
 * @param value Value to add
 */
template <typename T>
__aicore__ inline void UDMAAtomicAdd(const CommArgs& args, int target_rank,
                                      size_t flag_offset, T value) {
    if (!UDMAEnabled(args)) return;

    auto remote_flag = reinterpret_cast<__gm__ T*>(
        static_cast<T*>(args.peer_flag_ptrs[target_rank]) + flag_offset);
    aclshmemx_udma_atomic_add(remote_flag, value, target_rank);
}

/**
 * @brief Atomic fetch-and-add (returns old value)
 * @tparam T Integer type
 * @param args CommArgs with peer flag pointers
 * @param target_rank Destination rank ID
 * @param flag_offset Offset in remote rank's flag buffer (in T units)
 * @param value Value to add
 * @return Old value before addition
 */
template <typename T>
__aicore__ inline T UDMAAtomicFetchAdd(const CommArgs& args, int target_rank,
                                        size_t flag_offset, T value) {
    if (!UDMAEnabled(args)) return 0;

    auto remote_flag = reinterpret_cast<__gm__ T*>(
        static_cast<T*>(args.peer_flag_ptrs[target_rank]) + flag_offset);
    return aclshmemx_udma_atomic_fetch_add(remote_flag, value, target_rank);
}

/**
 * @brief Atomic compare-and-swap
 * @tparam T Integer type
 * @param args CommArgs with peer flag pointers
 * @param target_rank Destination rank ID
 * @param flag_offset Offset in remote rank's flag buffer (in T units)
 * @param expected Expected current value
 * @param desired New value to set if match
 * @return Old value (equals expected if swap succeeded)
 */
template <typename T>
__aicore__ inline T UDMAAtomicCompareSwap(const CommArgs& args, int target_rank,
                                           size_t flag_offset, T expected, T desired) {
    if (!UDMAEnabled(args)) return expected;

    auto remote_flag = reinterpret_cast<__gm__ T*>(
        static_cast<T*>(args.peer_flag_ptrs[target_rank]) + flag_offset);
    return aclshmemx_udma_atomic_compare_swap(remote_flag, expected, desired, target_rank);
}

} // namespace TileXR

#endif // TILEXR_UDMA_H
