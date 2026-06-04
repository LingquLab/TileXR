# CANN Version Migration Guide

This document records lessons learned and best practices when migrating TileXR across CANN versions.

## CANN 9.0.0 → 9.1.0 Migration

**Date**: 2026-05-25  
**Impact**: Build system, include paths, library locations

> Current note: the active TileXR UDMA implementation does not link shmem. The shmem section below is retained as historical context from an earlier UDMA design and for anyone comparing against the ignored `reference/shmem/` checkout; it is not a required step for building current `tile-comm`.

### Breaking Changes

#### 1. Include Path Structure

**CANN 9.0.0**:
```
${ASCEND_HOME_PATH}/${ARCH}-linux/
├── include/
│   ├── acl/
│   ├── aclnn/
│   └── runtime/
```

**CANN 9.1.0**:
```
${ASCEND_HOME_PATH}/${ARCH}-linux/
├── pkg_inc/                    # NEW: Root include directory
│   ├── acl/
│   ├── aclnn/
│   ├── runtime/
│   └── ...
```

**Fix**:
```cmake
# Old (9.0.0)
include_directories(${ASCEND_HOME_PATH}/${ARCH}-linux/include)

# New (9.1.0) - Must include pkg_inc root
include_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
)
```

**Symptoms**:
- `fatal error: acl/acl.h: No such file or directory`
- `fatal error: runtime/rt.h: No such file or directory`

#### 2. Library Location Changes

**CANN 9.0.0**:
```
${ASCEND_HOME_PATH}/${ARCH}-linux/
├── lib64/
│   ├── libascend_hal.so
│   ├── libascendcl.so
│   └── ...
```

**CANN 9.1.0**:
```
${ASCEND_HOME_PATH}/${ARCH}-linux/
├── lib64/
│   ├── libascendcl.so
│   └── ...
├── devlib/                     # NEW: Development libraries
│   ├── libascend_hal.so        # MOVED HERE
│   └── ...
```

**Fix**:
```cmake
# Old (9.0.0)
target_link_directories(tile-comm
    PUBLIC
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
)

# New (9.1.0) - Must include devlib
target_link_directories(tile-comm
    PUBLIC
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib  # ADD THIS
)
```

**Symptoms**:
- `/usr/bin/ld: cannot find -lascend_hal: No such file or directory`

#### 3. Historical shmem Library Changes

These notes applied to an earlier shmem-backed UDMA proposal.

**CANN 9.0.0 shmem**:
- Library name: `libaclshmem.so`
- Install path: `build/lib/`
- Public API: `aclshmem_instance_ctx` with `udma_info` field

**CANN 9.1.0 shmem**:
- Library name: `libshmem.so` (renamed)
- Install path: `install/shmem/lib/` (restructured)
- Public API: `aclshmem_instance_ctx` **without** `udma_info` field (removed)

**Historical fix**:
```cmake
# Old (9.0.0)
find_library(SHMEM_HOST_LIB aclshmem
    HINTS "${SHMEM_ROOT}/build/lib"
    REQUIRED)

# New (9.1.0)
find_library(SHMEM_HOST_LIB shmem
    HINTS "${SHMEM_ROOT}/install/shmem/lib"
    REQUIRED)
```

**Historical code fix** (requires custom shmem API):
```cpp
// Old (9.0.0) - Direct access
aclshmem_instance_ctx* ctx = aclshmemx_instance_ctx_get();
void* udma_info = ctx->udma_info;  // ❌ Field removed in 9.1.0

// New (9.1.0) - Custom API
void* udmaInfoPtr = nullptr;
size_t udmaInfoSize = 0;
aclshmemx_get_udma_info(&udmaInfoPtr, &udmaInfoSize);  // ✅ Custom API
```

### Migration Checklist

When upgrading CANN versions:

- [ ] Check `${ASCEND_HOME_PATH}` directory structure
- [ ] Update all `include_directories()` to use new paths
- [ ] Update all `target_link_directories()` to include new library locations
- [ ] Verify library names used by active targets have not changed
- [ ] Check for API changes in dependencies used by active targets
- [ ] Verify current `install/lib/libtile-comm.so` does not unexpectedly link shmem
- [ ] Test build on clean environment
- [ ] Update CLAUDE.md with version-specific notes
- [ ] Document breaking changes in this file

### Debugging Tips

#### Finding Missing Headers

```bash
# Search for header in CANN installation
find ${ASCEND_HOME_PATH} -name "acl.h" 2>/dev/null

# Check current include paths
grep -r "include_directories" CMakeLists.txt
```

#### Finding Missing Libraries

```bash
# Search for library in CANN installation
find ${ASCEND_HOME_PATH} -name "libascend_hal.so" 2>/dev/null

# Check what libraries are linked
ldd /path/to/libtile-comm.so

# Check current link directories
grep -r "target_link_directories" CMakeLists.txt
```

#### Verifying shmem is not an active TileXR dependency

```bash
ldd /path/to/libtile-comm.so | grep -i shmem || true
rg -n '#include "shmem\.h"|aclshmem|ACLSHMEM' src/comm src/include/tilexr_udma.h
```

Expected for the current implementation: no matches.

### Common Pitfalls

1. **Assuming backward compatibility**: CANN minor versions can have breaking changes
2. **Hardcoded paths**: Always use `${ASCEND_HOME_PATH}` and `${ARCH}` variables
3. **Incomplete path updates**: Must update both include AND link directories
4. **Confusing historical shmem notes with current build requirements**: current TileXR UDMA uses `src/comm/udma`, not a shmem link dependency
5. **Missing environment variables**: Always source `common_env.sh` before building

### Version Detection

Add version detection to CMakeLists.txt:

```cmake
# Detect CANN version
if(EXISTS "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc")
    message(STATUS "Detected CANN 9.1.0+ (pkg_inc structure)")
    set(CANN_INCLUDE_ROOT "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc")
    set(CANN_DEVLIB_DIR "${ASCEND_HOME_PATH}/${ARCH}-linux/devlib")
else()
    message(STATUS "Detected CANN 9.0.0 or earlier")
    set(CANN_INCLUDE_ROOT "${ASCEND_HOME_PATH}/${ARCH}-linux/include")
    set(CANN_DEVLIB_DIR "${ASCEND_HOME_PATH}/${ARCH}-linux/lib64")
endif()
```

## Future Considerations

### For CANN 9.2.0+

Monitor these areas for potential changes:

- AscendC SIMT API evolution
- HCCP/RA runtime symbol stability for TileXR-owned UDMA transport
- Runtime library reorganization
- New hardware support (Ascend 960, etc.)

### Maintaining Compatibility

**Strategy 1: Version-specific branches**
- Maintain separate branches for each CANN major version
- Cherry-pick bug fixes across branches

**Strategy 2: Conditional compilation**
- Use CMake version detection
- Maintain single codebase with `#ifdef` guards

**Current approach**: Version-specific branches (simpler for now)

## Related Documents

- [TileXR CLAUDE.md](../CLAUDE.md) - Build instructions and architecture
- [UDMA Integration Summary](./UDMA_INTEGRATION_SUMMARY.md) - Current TileXR UDMA architecture notes

## Changelog

- **2026-05-29**: Marked shmem-backed UDMA notes as historical; current TileXR-owned UDMA does not link shmem
- **2026-05-25**: Initial document, CANN 9.0.0 -> 9.1.0 migration notes
