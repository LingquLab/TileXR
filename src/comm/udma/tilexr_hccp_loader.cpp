/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_hccp_loader.h"

#include <dlfcn.h>

#include "mki/utils/log/log.h"

namespace TileXR {

template <typename T>
int TileXRHccpLoader::LoadSymbol(void* handle, T& dst, const char* primary, const char* fallback)
{
    dlerror();
    dst = reinterpret_cast<T>(dlsym(handle, primary));
    if (dst == nullptr && fallback != nullptr) {
        dst = reinterpret_cast<T>(dlsym(handle, fallback));
    }
    if (dst == nullptr) {
        MKI_LOG(WARN) << "TileXR UDMA failed to load symbol " << primary;
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }
    return TILEXR_HCCP_LOADER_SUCCESS;
}

int TileXRHccpLoader::Load()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded_) {
        return TILEXR_HCCP_LOADER_SUCCESS;
    }

    hcclV1Handle_ = dlopen("libhccl.so", RTLD_NOW);
    if (hcclV1Handle_ == nullptr) {
        MKI_LOG(WARN) << "TileXR UDMA failed to open libhccl.so: " << dlerror();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }
    hcclHandle_ = dlopen("libhccl_v2.so", RTLD_NOW);
    if (hcclHandle_ == nullptr) {
        MKI_LOG(WARN) << "TileXR UDMA failed to open libhccl_v2.so: " << dlerror();
        Unload();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }
    raHandle_ = dlopen("libra.so", RTLD_NOW);
    if (raHandle_ == nullptr) {
        MKI_LOG(WARN) << "TileXR UDMA failed to open libra.so: " << dlerror();
        Unload();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }
    tsdHandle_ = dlopen("libtsdclient.so", RTLD_NOW);
    if (tsdHandle_ == nullptr) {
        MKI_LOG(WARN) << "TileXR UDMA failed to open libtsdclient.so: " << dlerror();
        Unload();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }

    int ret = TILEXR_HCCP_LOADER_SUCCESS;
    ret |= LoadSymbol(hcclHandle_, RaInit, "RaInit", "ra_init");
    ret |= LoadSymbol(hcclHandle_, RaDeinit, "RaDeinit", "ra_deinit");
    ret |= LoadSymbol(tsdHandle_, TsdProcessOpen, "TsdProcessOpen", "tsd_process_open");
    ret |= LoadSymbol(tsdHandle_, TsdProcessClose, "TsdProcessClose", "tsd_process_close");
    ret |= LoadSymbol(hcclHandle_, RaGetDevEidInfoNum, "RaGetDevEidInfoNum", "ra_get_dev_eid_info_num");
    ret |= LoadSymbol(hcclHandle_, RaGetDevEidInfoList, "RaGetDevEidInfoList", "ra_get_dev_eid_info_list");
    ret |= LoadSymbol(hcclHandle_, RaCtxInit, "RaCtxInit", "ra_ctx_init");
    ret |= LoadSymbol(raHandle_, RaCtxChanCreate, "RaCtxChanCreate", "ra_ctx_chan_create");
    ret |= LoadSymbol(hcclHandle_, RaCtxCqCreate, "RaCtxCqCreate", "ra_ctx_cq_create");
    ret |= LoadSymbol(hcclHandle_, RaCtxQpCreate, "RaCtxQpCreate", "ra_ctx_qp_create");
    ret |= LoadSymbol(hcclHandle_, RaCtxTokenIdAlloc, "RaCtxTokenIdAlloc", "ra_ctx_token_id_alloc");
    ret |= LoadSymbol(hcclHandle_, RaCtxQpImport, "RaCtxQpImport", "ra_ctx_qp_import");
    ret |= LoadSymbol(hcclHandle_, RaCtxQpBind, "RaCtxQpBind", "ra_ctx_qp_bind");
    ret |= LoadSymbol(hcclHandle_, RaCtxLmemRegister, "RaCtxLmemRegister", "ra_ctx_lmem_register");
    ret |= LoadSymbol(hcclHandle_, RaCtxRmemImport, "RaCtxRmemImport", "ra_ctx_rmem_import");
    ret |= LoadSymbol(hcclHandle_, RaCtxRmemUnimport, "RaCtxRmemUnimport", "ra_ctx_rmem_unimport");
    ret |= LoadSymbol(hcclHandle_, RaCtxLmemUnregister, "RaCtxLmemUnregister", "ra_ctx_lmem_unregister");
    ret |= LoadSymbol(hcclHandle_, RaCtxQpUnbind, "RaCtxQpUnbind", "ra_ctx_qp_unbind");
    ret |= LoadSymbol(hcclHandle_, RaCtxQpUnimport, "RaCtxQpUnimport", "ra_ctx_qp_unimport");
    ret |= LoadSymbol(hcclHandle_, RaCtxTokenIdFree, "RaCtxTokenIdFree", "ra_ctx_token_id_free");
    ret |= LoadSymbol(hcclHandle_, RaCtxQpDestroy, "RaCtxQpDestroy", "ra_ctx_qp_destroy");
    ret |= LoadSymbol(hcclHandle_, RaCtxCqDestroy, "RaCtxCqDestroy", "ra_ctx_cq_destroy");
    ret |= LoadSymbol(raHandle_, RaCtxChanDestroy, "RaCtxChanDestroy", "ra_ctx_chan_destroy");
    ret |= LoadSymbol(hcclHandle_, RaCtxDeinit, "RaCtxDeinit", "ra_ctx_deinit");
    if (ret != TILEXR_HCCP_LOADER_SUCCESS) {
        Unload();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }

    loaded_ = true;
    return TILEXR_HCCP_LOADER_SUCCESS;
}

void TileXRHccpLoader::ResetSymbols()
{
    RaInit = nullptr;
    RaDeinit = nullptr;
    TsdProcessOpen = nullptr;
    TsdProcessClose = nullptr;
    RaGetDevEidInfoNum = nullptr;
    RaGetDevEidInfoList = nullptr;
    RaCtxInit = nullptr;
    RaCtxChanCreate = nullptr;
    RaCtxCqCreate = nullptr;
    RaCtxQpCreate = nullptr;
    RaCtxTokenIdAlloc = nullptr;
    RaCtxQpImport = nullptr;
    RaCtxQpBind = nullptr;
    RaCtxLmemRegister = nullptr;
    RaCtxRmemImport = nullptr;
    RaCtxRmemUnimport = nullptr;
    RaCtxLmemUnregister = nullptr;
    RaCtxQpUnbind = nullptr;
    RaCtxQpUnimport = nullptr;
    RaCtxTokenIdFree = nullptr;
    RaCtxQpDestroy = nullptr;
    RaCtxCqDestroy = nullptr;
    RaCtxChanDestroy = nullptr;
    RaCtxDeinit = nullptr;
}

void TileXRHccpLoader::Unload()
{
    ResetSymbols();
    if (tsdHandle_ != nullptr) {
        dlclose(tsdHandle_);
        tsdHandle_ = nullptr;
    }
    if (raHandle_ != nullptr) {
        dlclose(raHandle_);
        raHandle_ = nullptr;
    }
    if (hcclHandle_ != nullptr) {
        dlclose(hcclHandle_);
        hcclHandle_ = nullptr;
    }
    if (hcclV1Handle_ != nullptr) {
        dlclose(hcclV1Handle_);
        hcclV1Handle_ = nullptr;
    }
    loaded_ = false;
}

bool TileXRHccpLoader::Loaded() const
{
    return loaded_;
}

} // namespace TileXR
