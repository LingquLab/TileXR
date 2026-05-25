# shmem Integration Guide

This document describes the custom shmem modifications required for TileXR UDMA integration.

## Overview

TileXR requires access to UDMA (User-space Direct Memory Access) queue information for device-side kernels. The standard shmem library (CANN 9.1.0) does not expose this information through public APIs. This guide documents the custom API added to shmem.

## Custom API

### `aclshmemx_get_udma_info()`

**Purpose**: Retrieve UDMA device memory pointer and size for use in device kernels.

**Signature**:
```cpp
int aclshmemx_get_udma_info(void** udma_info_ptr, size_t* udma_info_size);
```

**Parameters**:
- `udma_info_ptr` [out]: Pointer to receive device memory address of UDMA info structure
- `udma_info_size` [out]: Pointer to receive size of UDMA info structure in bytes

**Returns**:
- `ACLSHMEM_SUCCESS` (0) on success
- `ACLSHMEM_INVALID_PARAM` if parameters are null
- `ACLSHMEM_INNER_ERROR` if shmem not initialized or UDMA not available

**Example**:
```cpp
void* udmaInfoPtr = nullptr;
size_t udmaInfoSize = 0;
int ret = aclshmemx_get_udma_info(&udmaInfoPtr, &udmaInfoSize);
if (ret == ACLSHMEM_SUCCESS) {
    // udmaInfoPtr points to device memory containing ACLSHMEMAIVUDMAInfo
    // Pass this pointer to device kernels via kernel arguments
}
```

## Implementation Details

### Files Modified

1. **`include/host/init/shmem_host_init.h`**
   - Added public API declaration
   - Added documentation comments

2. **`src/host/init/shmem_init.cpp`**
   - Implemented `aclshmemx_get_udma_info()`
   - Returns `g_state.qp_info` (device memory pointer)
   - Calculates size based on rank count and structure layout

### Data Structure

The UDMA info structure in device memory:

```cpp
struct ACLSHMEMAIVUDMAInfo {
    uint32_t qpNum;   // Number of QPs per connection (typically 1)
    uint64_t sqPtr;   // Send queue array pointer [PE_NUM][qpNum]
    uint64_t rqPtr;   // Receive queue array pointer [PE_NUM][qpNum]
    uint64_t scqPtr;  // Send completion queue array [PE_NUM][qpNum]
    uint64_t rcqPtr;  // Receive completion queue array [PE_NUM][qpNum]
    uint64_t memPtr;  // Memory region array [MAX_PE_NUM]
};
```

**Memory layout**:
```
Device Memory:
┌─────────────────────────────────┐
│ ACLSHMEMAIVUDMAInfo (header)    │ sizeof(ACLSHMEMAIVUDMAInfo)
├─────────────────────────────────┤
│ ACLSHMEMUDMAWQCtx[N] (SQ)       │ N = rankCount * qpNum
├─────────────────────────────────┤
│ ACLSHMEMUDMAWQCtx[N] (RQ)       │ N = rankCount * qpNum
├─────────────────────────────────┤
│ ACLSHMEMUDMACqCtx[N] (SCQ)      │ N = rankCount * qpNum
├─────────────────────────────────┤
│ ACLSHMEMUDMACqCtx[N] (RCQ)      │ N = rankCount * qpNum
├─────────────────────────────────┤
│ ACLSHMEMUBmemInfo[N] (MEM)      │ N = rankCount
└─────────────────────────────────┘
```

### Initialization Flow

```
Host Application
    │
    ├─> aclshmemx_get_uniqueid()
    │
    ├─> aclshmemx_set_attr_uniqueid_args()
    │       └─> Set option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA
    │
    ├─> aclshmemx_init_attr()
    │       └─> DeviceJettyManager::FillUdmaInfo()
    │               └─> Allocates device memory
    │               └─> Fills UDMA queue contexts
    │               └─> Stores pointer in g_state.qp_info
    │
    ├─> aclshmemx_get_udma_info()  ← NEW API
    │       └─> Returns g_state.qp_info
    │
    └─> Pass pointer to device kernel
            └─> Kernel uses UDMA queues for communication
```

## Building Modified shmem

### Prerequisites

- CANN 9.1.0 or later
- GCC 13.3.0 or compatible
- CMake 3.16+

### Build Steps

```bash
cd /path/to/shmem
source /path/to/cann/set_env.sh

# Build with UDMA support (Ascend950 required)
bash scripts/build.sh -soc_type Ascend950

# Output: install/shmem/lib/libshmem.so
```

### Build Options

- `-soc_type Ascend950`: Enables UDMA support (required)
- `-package ON`: Create installation package
- `-all`: Build library, examples, and tests

### Verification

```bash
# Check if API is exported
nm -D install/shmem/lib/libshmem.so | grep aclshmemx_get_udma_info

# Expected output:
# 00000000001a2b40 T aclshmemx_get_udma_info
```

## Integration with TileXR

### CMake Configuration

```cmake
# Find shmem library
set(SHMEM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/shmem")
find_library(SHMEM_HOST_LIB shmem
    HINTS "${SHMEM_ROOT}/install/shmem/lib"
    REQUIRED)

# Include shmem headers
target_include_directories(tile-comm
    PRIVATE
    ${SHMEM_ROOT}/install/shmem/include
)

# Link shmem library
target_link_libraries(tile-comm ${SHMEM_HOST_LIB})
```

### Runtime Configuration

```bash
# Add shmem library to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/shmem/install/shmem/lib:$LD_LIBRARY_PATH
```

### Code Example

```cpp
#include "host/init/shmem_host_init.h"

class TileXRComm {
public:
    int InitUDMA() {
        // Step 1: Get unique ID
        aclshmemx_uniqueid_t shmemUid;
        int ret = aclshmemx_get_uniqueid(&shmemUid);
        if (ret != ACLSHMEM_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }

        // Step 2: Set attributes
        aclshmemx_init_attr_t shmemAttr = {};
        ret = aclshmemx_set_attr_uniqueid_args(
            rank_, rankSize_, 100 * 1024 * 1024,
            &shmemUid, &shmemAttr
        );
        if (ret != ACLSHMEM_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }

        // Step 3: Enable UDMA engine
        shmemAttr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;

        // Step 4: Initialize shmem
        ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &shmemAttr);
        if (ret != ACLSHMEM_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }

        // Step 5: Get UDMA info (NEW API)
        void* udmaInfoPtr = nullptr;
        size_t udmaInfoSize = 0;
        ret = aclshmemx_get_udma_info(&udmaInfoPtr, &udmaInfoSize);
        if (ret != ACLSHMEM_SUCCESS) {
            aclshmem_finalize();
            return TILEXR_ERROR_INTERNAL;
        }

        // Step 6: Store pointer for device kernels
        udmaInfoDev_ = reinterpret_cast<uint8_t*>(udmaInfoPtr);
        commArgs_.udmaInfoPtr = udmaInfoDev_;
        commArgs_.extraFlag |= ExtraFlag::UDMA;

        return TILEXR_SUCCESS;
    }

    ~TileXRComm() {
        if (udmaInfoDev_ != nullptr) {
            aclshmem_finalize();  // Cleanup shmem resources
            udmaInfoDev_ = nullptr;
        }
    }

private:
    uint8_t* udmaInfoDev_ = nullptr;
};
```

## Maintenance

### Updating shmem

When updating to a new shmem version:

1. Check if `aclshmemx_get_udma_info()` still exists
2. Verify `g_state.qp_info` is still populated
3. Test UDMA initialization with new version
4. Update this document if API changes

### Upstreaming

**Goal**: Contribute this API to upstream shmem repository.

**Status**: Custom patch, not yet upstreamed.

**Rationale for upstream**:
- Enables external libraries to use shmem's UDMA infrastructure
- Provides controlled access to device-side communication resources
- Maintains encapsulation (returns opaque pointer, not internal structures)

**Proposed upstream API** (same as current):
```cpp
/**
 * @brief Get UDMA information pointer for device-side communication.
 *
 * This function returns the device memory pointer to UDMA queue pair information
 * that can be used by device kernels for direct UDMA operations.
 *
 * @param udma_info_ptr [out] Pointer to receive device memory address
 * @param udma_info_size [out] Pointer to receive size in bytes
 * @return 0 on success, error code on failure
 */
ACLSHMEM_HOST_API int aclshmemx_get_udma_info(void** udma_info_ptr, size_t* udma_info_size);
```

## Troubleshooting

### API Not Found

**Symptom**:
```
undefined reference to `aclshmemx_get_udma_info'
```

**Solution**:
- Verify shmem was built with custom patch applied
- Check `nm -D libshmem.so | grep aclshmemx_get_udma_info`
- Rebuild shmem if necessary

### UDMA Not Available

**Symptom**:
```
aclshmemx_get_udma_info() returns ACLSHMEM_INNER_ERROR
```

**Possible causes**:
1. shmem not initialized: Call `aclshmemx_init_attr()` first
2. UDMA not enabled: Set `option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA`
3. Hardware not supported: UDMA requires Ascend 950 or later
4. Build configuration: shmem must be built with `-soc_type Ascend950`

**Debug**:
```cpp
// Check initialization status
int status = aclshmemx_init_status();
if (status != ACLSHMEM_STATUS_IS_INITIALIZED) {
    // shmem not initialized
}

// Check if qp_info is set (internal debugging)
// Note: This requires access to g_state, not available in public API
```

### Device Kernel Crashes

**Symptom**: Kernel crashes when accessing UDMA info pointer

**Possible causes**:
1. Pointer not passed correctly to kernel
2. Device memory not accessible from kernel context
3. Structure layout mismatch between host and device

**Debug**:
```cpp
// Verify pointer is valid device memory
aclrtPointerAttributes attr;
aclrtPointerGetAttributes(&attr, udmaInfoPtr);
// attr.memoryType should be ACL_MEMTYPE_DEVICE

// Log pointer value
printf("UDMA info ptr: %p, size: %zu\n", udmaInfoPtr, udmaInfoSize);
```

## Version History

- **v1.0 (2026-05-25)**: Initial implementation for TileXR UDMA integration
  - Added `aclshmemx_get_udma_info()` API
  - Based on shmem commit b79bda3
  - CANN 9.1.0 compatibility

## Related Documents

- [CANN Version Migration Guide](./CANN_VERSION_MIGRATION.md)
- [TileXR CLAUDE.md](../CLAUDE.md)
- [UDMA Integration Report](/tmp/udma_integration_summary.md)

## Contact

For questions about this integration:
- Check TileXR issues: https://gitcode.com/LingquLab/TileXR/issues
- Review shmem documentation: `/path/to/shmem/docs/`
