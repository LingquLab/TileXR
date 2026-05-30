/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_COMPAT_H
#define TILEXR_SDMA_COMPAT_H

#include "comm_args.h"
#include "tilexr_sdma_types.h"

#if defined(__has_include)
#if __has_include("tilexr_sdma_config.h")
#include "tilexr_sdma_config.h"
#endif
#endif

#ifndef TILEXR_HAVE_PTO_SDMA
#define TILEXR_HAVE_PTO_SDMA 0
#endif

#if TILEXR_ASCENDC_AICORE_COMPILE && defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA
#ifndef PTO_COMM_NOT_SUPPORTED
#define PTO_COMM_NOT_SUPPORTED 1
#endif
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#endif

#endif // TILEXR_SDMA_COMPAT_H
