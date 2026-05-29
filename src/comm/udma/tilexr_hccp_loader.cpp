/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_hccp_loader.h"

#include <dlfcn.h>

#include "tilexr_log.h"

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
        TILEXR_LOG(WARN) << "TileXR UDMA failed to load symbol " << primary;
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

    raHandle_ = dlopen("libra.so", RTLD_NOW);
    if (raHandle_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXR UDMA failed to open libra.so: " << dlerror();
        Unload();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }
    tsdHandle_ = dlopen("libtsdclient.so", RTLD_NOW);
    if (tsdHandle_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXR UDMA failed to open libtsdclient.so: " << dlerror();
        Unload();
        return TILEXR_HCCP_LOADER_NOT_FOUND;
    }

    int ret = TILEXR_HCCP_LOADER_SUCCESS;
    ret |= LoadSymbol(raHandle_, RaInit, "RaInit", "ra_init");
    ret |= LoadSymbol(raHandle_, RaDeinit, "RaDeinit", "ra_deinit");
    ret |= LoadSymbol(tsdHandle_, TsdProcessOpen, "TsdProcessOpen", "tsd_process_open");
    ret |= LoadSymbol(tsdHandle_, TsdProcessClose, "TsdProcessClose", "tsd_process_close");
    ret |= LoadSymbol(raHandle_, RaGetDevEidInfoNum, "RaGetDevEidInfoNum", "ra_get_dev_eid_info_num");
    ret |= LoadSymbol(raHandle_, RaGetDevEidInfoList, "RaGetDevEidInfoList", "ra_get_dev_eid_info_list");
    ret |= LoadSymbol(raHandle_, RaCtxInit, "RaCtxInit", "ra_ctx_init");
    ret |= LoadSymbol(raHandle_, RaCtxChanCreate, "RaCtxChanCreate", "ra_ctx_chan_create");
    ret |= LoadSymbol(raHandle_, RaCtxCqCreate, "RaCtxCqCreate", "ra_ctx_cq_create");
    ret |= LoadSymbol(raHandle_, RaCtxQpCreate, "RaCtxQpCreate", "ra_ctx_qp_create");
    ret |= LoadSymbol(raHandle_, RaCtxTokenIdAlloc, "RaCtxTokenIdAlloc", "ra_ctx_token_id_alloc");
    ret |= LoadSymbol(raHandle_, RaCtxQpImport, "RaCtxQpImport", "ra_ctx_qp_import");
    ret |= LoadSymbol(raHandle_, RaCtxQpBind, "RaCtxQpBind", "ra_ctx_qp_bind");
    ret |= LoadSymbol(raHandle_, RaCtxLmemRegister, "RaCtxLmemRegister", "ra_ctx_lmem_register");
    ret |= LoadSymbol(raHandle_, RaCtxRmemImport, "RaCtxRmemImport", "ra_ctx_rmem_import");
    ret |= LoadSymbol(raHandle_, RaCtxRmemUnimport, "RaCtxRmemUnimport", "ra_ctx_rmem_unimport");
    ret |= LoadSymbol(raHandle_, RaCtxLmemUnregister, "RaCtxLmemUnregister", "ra_ctx_lmem_unregister");
    ret |= LoadSymbol(raHandle_, RaCtxQpUnbind, "RaCtxQpUnbind", "ra_ctx_qp_unbind");
    ret |= LoadSymbol(raHandle_, RaCtxQpUnimport, "RaCtxQpUnimport", "ra_ctx_qp_unimport");
    ret |= LoadSymbol(raHandle_, RaCtxTokenIdFree, "RaCtxTokenIdFree", "ra_ctx_token_id_free");
    ret |= LoadSymbol(raHandle_, RaCtxQpDestroy, "RaCtxQpDestroy", "ra_ctx_qp_destroy");
    ret |= LoadSymbol(raHandle_, RaCtxCqDestroy, "RaCtxCqDestroy", "ra_ctx_cq_destroy");
    ret |= LoadSymbol(raHandle_, RaCtxChanDestroy, "RaCtxChanDestroy", "ra_ctx_chan_destroy");
    ret |= LoadSymbol(raHandle_, RaCtxDeinit, "RaCtxDeinit", "ra_ctx_deinit");
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
    loaded_ = false;
}

bool TileXRHccpLoader::Loaded() const
{
    return loaded_;
}

} // namespace TileXR
