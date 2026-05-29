/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_HCCP_LOADER_H
#define TILEXR_HCCP_LOADER_H

#include <mutex>

#include "tilexr_hccp_defs.h"

namespace TileXR {

constexpr int TILEXR_HCCP_LOADER_SUCCESS = 0;
constexpr int TILEXR_HCCP_LOADER_NOT_FOUND = -7;

using RaInitFunc = int (*)(RaInitConfig*);
using RaDeinitFunc = int (*)(RaInitConfig*);
using TsdProcessOpenFunc = uint32_t (*)(const uint32_t, ProcOpenArgs*);
using TsdProcessCloseFunc = uint32_t (*)(const uint32_t, const pid_t);
using RaGetDevEidInfoNumFunc = int (*)(RaInfo, unsigned int*);
using RaGetDevEidInfoListFunc = int (*)(RaInfo, DevEidInfo[], unsigned int*);
using RaCtxInitFunc = int (*)(CtxInitCfg*, CtxInitAttr*, void**);
using RaCtxChanCreateFunc = int (*)(void*, ChanInfoT*, void**);
using RaCtxCqCreateFunc = int (*)(void*, CqInfoT*, void**);
using RaCtxQpCreateFunc = int (*)(void*, QpCreateAttr*, QpCreateInfo*, void**);
using RaCtxTokenIdAllocFunc = int (*)(void*, HccpTokenId*, void**);
using RaCtxQpImportFunc = int (*)(void*, QpImportInfoT*, void**);
using RaCtxQpBindFunc = int (*)(void*, void*);
using RaCtxLmemRegisterFunc = int (*)(void*, MrRegInfoT*, void**);
using RaCtxRmemImportFunc = int (*)(void*, MrImportInfoT*, void**);
using RaCtxRmemUnimportFunc = int (*)(void*, void*);
using RaCtxLmemUnregisterFunc = int (*)(void*, void*);
using RaCtxQpUnbindFunc = int (*)(void*);
using RaCtxQpUnimportFunc = int (*)(void*, void*);
using RaCtxTokenIdFreeFunc = int (*)(void*, void*);
using RaCtxQpDestroyFunc = int (*)(void*);
using RaCtxCqDestroyFunc = int (*)(void*, void*);
using RaCtxChanDestroyFunc = int (*)(void*, void*);
using RaCtxDeinitFunc = int (*)(void*);

class TileXRHccpLoader {
public:
    int Load();
    void Unload();
    bool Loaded() const;

    RaInitFunc RaInit = nullptr;
    RaDeinitFunc RaDeinit = nullptr;
    TsdProcessOpenFunc TsdProcessOpen = nullptr;
    TsdProcessCloseFunc TsdProcessClose = nullptr;
    RaGetDevEidInfoNumFunc RaGetDevEidInfoNum = nullptr;
    RaGetDevEidInfoListFunc RaGetDevEidInfoList = nullptr;
    RaCtxInitFunc RaCtxInit = nullptr;
    RaCtxChanCreateFunc RaCtxChanCreate = nullptr;
    RaCtxCqCreateFunc RaCtxCqCreate = nullptr;
    RaCtxQpCreateFunc RaCtxQpCreate = nullptr;
    RaCtxTokenIdAllocFunc RaCtxTokenIdAlloc = nullptr;
    RaCtxQpImportFunc RaCtxQpImport = nullptr;
    RaCtxQpBindFunc RaCtxQpBind = nullptr;
    RaCtxLmemRegisterFunc RaCtxLmemRegister = nullptr;
    RaCtxRmemImportFunc RaCtxRmemImport = nullptr;
    RaCtxRmemUnimportFunc RaCtxRmemUnimport = nullptr;
    RaCtxLmemUnregisterFunc RaCtxLmemUnregister = nullptr;
    RaCtxQpUnbindFunc RaCtxQpUnbind = nullptr;
    RaCtxQpUnimportFunc RaCtxQpUnimport = nullptr;
    RaCtxTokenIdFreeFunc RaCtxTokenIdFree = nullptr;
    RaCtxQpDestroyFunc RaCtxQpDestroy = nullptr;
    RaCtxCqDestroyFunc RaCtxCqDestroy = nullptr;
    RaCtxChanDestroyFunc RaCtxChanDestroy = nullptr;
    RaCtxDeinitFunc RaCtxDeinit = nullptr;

private:
    template <typename T>
    int LoadSymbol(void* handle, T& dst, const char* primary, const char* fallback);
    void ResetSymbols();

    void* hcclV1Handle_ = nullptr;
    void* hcclHandle_ = nullptr;
    void* raHandle_ = nullptr;
    void* tsdHandle_ = nullptr;
    bool loaded_ = false;
    std::mutex mutex_;
};

} // namespace TileXR

#endif // TILEXR_HCCP_LOADER_H
