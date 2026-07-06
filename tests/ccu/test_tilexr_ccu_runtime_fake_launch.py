#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMM_DIR = REPO_ROOT / "src" / "comm"
INCLUDE_DIR = REPO_ROOT / "src" / "include"
RUNTIME_SOURCE = COMM_DIR / "ccu" / "tilexr_ccu_runtime.cpp"
ORCHESTRATOR_SOURCE = COMM_DIR / "ccu" / "tilexr_ccu_direct_orchestrator.cpp"
REPOSITORY_SOURCE = COMM_DIR / "ccu" / "tilexr_ccu_repository.cpp"
COMM_WRAP_SOURCE = COMM_DIR / "comm_wrap.cpp"


class TileXRCcuRuntimeFakeLaunchTest(unittest.TestCase):
    def compile_and_run(self, code: str):
        compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("no local C++ compiler found")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_dir = temp_path / "runtime"
            runtime_dir.mkdir(parents=True)
            acl_dir = temp_path / "acl"
            acl_dir.mkdir(parents=True)
            (acl_dir / "acl_rt.h").write_text(
                textwrap.dedent(
                    r"""
                    #ifndef TILEXR_TEST_FAKE_ACL_RT_H
                    #define TILEXR_TEST_FAKE_ACL_RT_H

                    #include <stddef.h>

                    #define ACL_SUCCESS 0
                    #define ACL_MEM_MALLOC_HUGE_FIRST 0
                    #define ACL_MEMCPY_HOST_TO_DEVICE 0
                    #define ACL_MEMCPY_DEVICE_TO_HOST 1
                    #define ACL_MEM_TYPE_HIGH_BAND_WIDTH 0
                    #define ACL_RT_MEM_ATTR_MODULE_ID 0

                    typedef int aclError;
                    typedef int aclrtMemMallocPolicy;
                    typedef union aclrtMallocAttrValue {
                        unsigned int moduleId;
                    } aclrtMallocAttrValue;
                    typedef struct aclrtMallocAttribute {
                        int attr;
                        aclrtMallocAttrValue value;
                    } aclrtMallocAttribute;
                    typedef struct aclrtMallocConfig {
                        aclrtMallocAttribute* attrs;
                        size_t attrCount;
                    } aclrtMallocConfig;

                    extern "C" aclError aclrtSetDevice(int deviceId);
                    extern "C" aclError aclrtMalloc(void** devPtr, size_t size, int policy);
                    static inline aclError aclrtMallocWithCfg(
                        void** devPtr, size_t size, int memoryType, aclrtMallocConfig* cfg)
                    {
                        (void)memoryType;
                        (void)cfg;
                        return aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
                    }
                    extern "C" aclError aclrtMemcpy(
                        void* dst, size_t destMax, const void* src, size_t count, int kind);
                    extern "C" aclError aclrtFree(void* devPtr);

                    #endif
                    """
                ),
                encoding="utf-8",
            )
            (runtime_dir / "kernel.h").write_text(
                textwrap.dedent(
                    r"""
                    #ifndef TILEXR_TEST_FAKE_RUNTIME_KERNEL_H
                    #define TILEXR_TEST_FAKE_RUNTIME_KERNEL_H

                    #include <stdint.h>

                    #define RT_CCU_SQE_ARGS_LEN 13U
                    #define RT_CCU_INST_CNT_INVALID 0U
                    #define RT_CCU_INST_START_MAX 65535U
                    #define RT_ERROR_NONE 0

                    typedef int32_t rtError_t;
                    typedef void* rtStream_t;

                    typedef struct rtCcuTaskInfo {
                        uint8_t dieId;
                        uint8_t missionId;
                        uint16_t timeout;
                        uint16_t instStartId;
                        uint16_t instCnt;
                        uint32_t key;
                        uint32_t argSize;
                        uint64_t args[RT_CCU_SQE_ARGS_LEN];
                    } rtCcuTaskInfo_t;

                    extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t stream);

                    #endif
                    """
                ),
                encoding="utf-8",
            )

            test_cpp = temp_path / "test.cpp"
            test_cpp.write_text(code, encoding="utf-8")
            test_bin = temp_path / "test_fake_launch"
            subprocess.run(
                [
                    compiler,
                    "-std=c++14",
                    "-I",
                    str(temp_path),
                    "-I",
                    str(INCLUDE_DIR),
                    "-I",
                    str(COMM_DIR),
                    str(test_cpp),
                    str(RUNTIME_SOURCE),
                    "-o",
                    str(test_bin),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            return subprocess.run([str(test_bin)], cwd=REPO_ROOT, check=False, text=True, capture_output=True)

    def compile_and_run_public_prepared_handle(self, code: str):
        compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("no local C++ compiler found")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_dir = temp_path / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.h").write_text(
                textwrap.dedent(
                    r"""
                    #ifndef TILEXR_TEST_FAKE_RUNTIME_KERNEL_H
                    #define TILEXR_TEST_FAKE_RUNTIME_KERNEL_H

                    #include <stdint.h>

                    #define RT_CCU_SQE_ARGS_LEN 13U
                    #define RT_CCU_INST_CNT_INVALID 0U
                    #define RT_CCU_INST_START_MAX 65535U
                    #define RT_ERROR_NONE 0

                    typedef int32_t rtError_t;
                    typedef void* rtStream_t;

                    typedef struct rtCcuTaskInfo {
                        uint8_t dieId;
                        uint8_t missionId;
                        uint16_t timeout;
                        uint16_t instStartId;
                        uint16_t instCnt;
                        uint32_t key;
                        uint32_t argSize;
                        uint64_t args[RT_CCU_SQE_ARGS_LEN];
                    } rtCcuTaskInfo_t;

                    extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t stream);

                    #endif
                    """
                ),
                encoding="utf-8",
            )
            acl_dir = temp_path / "acl"
            acl_dir.mkdir(parents=True)
            (acl_dir / "acl_rt.h").write_text(
                textwrap.dedent(
                    r"""
                    #ifndef TILEXR_TEST_FAKE_ACL_RT_H
                    #define TILEXR_TEST_FAKE_ACL_RT_H

                    #include <stddef.h>

                    #define ACL_SUCCESS 0
                    #define ACL_MEM_MALLOC_HUGE_FIRST 0
                    #define ACL_MEMCPY_HOST_TO_DEVICE 0
                    #define ACL_MEMCPY_DEVICE_TO_HOST 1
                    #define ACL_MEM_TYPE_HIGH_BAND_WIDTH 0
                    #define ACL_RT_MEM_ATTR_MODULE_ID 0

                    typedef int aclError;
                    typedef int aclrtMemMallocPolicy;
                    typedef union aclrtMallocAttrValue {
                        unsigned int moduleId;
                    } aclrtMallocAttrValue;
                    typedef struct aclrtMallocAttribute {
                        int attr;
                        aclrtMallocAttrValue value;
                    } aclrtMallocAttribute;
                    typedef struct aclrtMallocConfig {
                        aclrtMallocAttribute* attrs;
                        size_t attrCount;
                    } aclrtMallocConfig;

                    extern "C" aclError aclrtSetDevice(int deviceId);
                    extern "C" aclError aclrtMalloc(void** devPtr, size_t size, int policy);
                    static inline aclError aclrtMallocWithCfg(
                        void** devPtr, size_t size, int memoryType, aclrtMallocConfig* cfg)
                    {
                        (void)memoryType;
                        (void)cfg;
                        return aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
                    }
                    extern "C" aclError aclrtMemcpy(
                        void* dst, size_t destMax, const void* src, size_t count, int kind);
                    extern "C" aclError aclrtFree(void* devPtr);

                    #endif
                    """
                ),
                encoding="utf-8",
            )

            test_cpp = temp_path / "test_public_prepared.cpp"
            test_cpp.write_text(code, encoding="utf-8")
            test_bin = temp_path / "test_public_prepared"
            subprocess.run(
                [
                    compiler,
                    "-std=c++14",
                    "-DTILEXR_CCU_TESTING=1",
                    "-DTILEXR_LOG_DISABLE_SPDLOG=1",
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-I",
                    str(temp_path),
                    "-I",
                    str(INCLUDE_DIR),
                    "-I",
                    str(COMM_DIR),
                    str(test_cpp),
                    str(COMM_WRAP_SOURCE),
                    str(ORCHESTRATOR_SOURCE),
                    str(REPOSITORY_SOURCE),
                    str(RUNTIME_SOURCE),
                    "-Wl,--gc-sections",
                    "-o",
                    str(test_bin),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            return subprocess.run([str(test_bin)], cwd=REPO_ROOT, check=False, text=True, capture_output=True)

    def test_submit_task_copies_tilexr_fields_to_runtime_task_info(self):
        code = textwrap.dedent(
            r"""
            #include "ccu/tilexr_ccu_runtime.h"
            #include "tilexr_types.h"

            #include <cstdint>
            #include <cstdio>
            #include <cstring>

            #include <runtime/kernel.h>

            namespace {
            rtCcuTaskInfo_t g_capturedTask {};
            rtStream_t g_capturedStream = nullptr;
            int g_launchCount = 0;
            }

            extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t stream)
            {
                ++g_launchCount;
                g_capturedStream = stream;
                std::memcpy(&g_capturedTask, taskInfo, sizeof(g_capturedTask));
                return RT_ERROR_NONE;
            }

            int main()
            {
                TileXR::TileXRCcuTask task {};
                task.dieId = 3;
                task.missionId = 7;
                task.timeout = 68;
                task.instStartId = 1024;
                task.instCnt = 13;
                task.key = 0x05ab1234U;
                task.argSize = TileXR::TILEXR_CCU_SQE_ARGS_LEN;
                for (uint32_t i = 0; i < TileXR::TILEXR_CCU_SQE_ARGS_LEN; ++i) {
                    task.args[i] = 0x1000000000000000ULL + i;
                }

                void* stream = reinterpret_cast<void*>(0x12345678ULL);
                const int ret = TileXR::TileXRCcuSubmitTask(task, stream);
                if (ret != TileXR::TILEXR_SUCCESS) {
                    std::printf("unexpected submit ret=%d\n", ret);
                    return 1;
                }
                if (g_launchCount != 1 || g_capturedStream != stream) {
                    std::printf("launchCount=%d capturedStream=%p\n", g_launchCount, g_capturedStream);
                    return 2;
                }
                if (g_capturedTask.dieId != task.dieId || g_capturedTask.missionId != task.missionId ||
                    g_capturedTask.timeout != task.timeout || g_capturedTask.instStartId != task.instStartId ||
                    g_capturedTask.instCnt != task.instCnt || g_capturedTask.key != task.key ||
                    g_capturedTask.argSize != task.argSize) {
                    std::printf("runtime scalar field mismatch\n");
                    return 3;
                }
                for (uint32_t i = 0; i < TileXR::TILEXR_CCU_SQE_ARGS_LEN; ++i) {
                    if (g_capturedTask.args[i] != task.args[i]) {
                        std::printf("arg[%u] mismatch\n", i);
                        return 4;
                    }
                }
                return 0;
            }
            """
        )

        result = self.compile_and_run(code)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_submit_report_records_final_runtime_task(self):
        code = textwrap.dedent(
            r"""
            #include "ccu/tilexr_ccu_runtime.h"
            #include "tilexr_types.h"

            #include <cstdint>
            #include <cstdio>
            #include <cstring>

            #include <runtime/kernel.h>

            namespace {
            rtCcuTaskInfo_t g_capturedTask {};
            }

            extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t)
            {
                std::memcpy(&g_capturedTask, taskInfo, sizeof(g_capturedTask));
                return RT_ERROR_NONE;
            }

            int main()
            {
                TileXR::TileXRCcuTask task {};
                task.dieId = 3;
                task.missionId = 6;
                task.timeout = 68;
                task.instStartId = 489;
                task.instCnt = 2;
                task.key = 0x059b0f03U;
                task.argSize = TileXR::TILEXR_CCU_SQE_ARGS_LEN;
                task.args[0] = 0x1111ULL;
                task.args[1] = 0x2222ULL;

                TileXR::TileXRCcuRuntimeSubmitReport report {};
                const int ret = TileXR::TileXRCcuSubmitTaskWithReport(
                    task, reinterpret_cast<void*>(0x1ULL), &report);
                if (ret != TileXR::TILEXR_SUCCESS) {
                    std::printf("submit ret=%d\n", ret);
                    return 1;
                }
                if (!report.finalTaskCaptured) {
                    std::printf("final task was not captured\n");
                    return 2;
                }
                if (report.finalTask.dieId != task.dieId || report.finalTask.timeout != task.timeout ||
                    report.finalTask.argSize != task.argSize || report.finalTask.args[0] != 0x1111ULL ||
                    report.finalTask.args[1] != 0x2222ULL) {
                    std::printf("final task mismatch die=%u timeout=%u argSize=%u arg0=0x%llx arg1=0x%llx\n",
                        static_cast<unsigned>(report.finalTask.dieId),
                        static_cast<unsigned>(report.finalTask.timeout),
                        static_cast<unsigned>(report.finalTask.argSize),
                        static_cast<unsigned long long>(report.finalTask.args[0]),
                        static_cast<unsigned long long>(report.finalTask.args[1]));
                    return 3;
                }
                if (g_capturedTask.dieId != report.finalTask.dieId ||
                    g_capturedTask.timeout != report.finalTask.timeout ||
                    g_capturedTask.argSize != report.finalTask.argSize ||
                    g_capturedTask.args[0] != report.finalTask.args[0]) {
                    std::printf("captured runtime task differs from report\n");
                    return 4;
                }
                return 0;
            }
            """
        )

        result = self.compile_and_run(code)
        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_submit_task_maps_runtime_launch_failure_to_mkirt_error(self):
        code = textwrap.dedent(
            r"""
            #include "ccu/tilexr_ccu_runtime.h"
            #include "tilexr_types.h"

            #include <runtime/kernel.h>

            extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t*, rtStream_t)
            {
                return 507000;
            }

            int main()
            {
                TileXR::TileXRCcuTask task {};
                task.dieId = 1;
                task.missionId = 2;
                task.instStartId = 8;
                task.instCnt = 1;
                task.key = 0x1234U;
                task.argSize = 1;
                task.args[0] = 0xfeedULL;

                const int ret = TileXR::TileXRCcuSubmitTask(task, reinterpret_cast<void*>(0x1ULL));
                return ret == TileXR::TILEXR_ERROR_MKIRT ? 0 : 1;
            }
            """
        )

        result = self.compile_and_run(code)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_public_prepared_handle_submits_batch_through_runtime_launch(self):
        code = textwrap.dedent(
            r"""
            #include "tilexr_api.h"
            #include "tilexr_types.h"

            #include <cstdint>
            #include <cstdio>
            #include <cstring>

            #include <acl/acl_rt.h>
            #include <runtime/kernel.h>

            extern "C" TileXRDirectCcuPreparedTasksPtr TileXRDirectCcuCreatePreparedForTest(
                const TileXRDirectCcuTaskInfo* tasks, uint32_t taskCount);

            namespace {
            rtCcuTaskInfo_t g_tasks[4] {};
            rtStream_t g_streams[4] {};
            int g_launchCount = 0;
            int g_failOnCall = 0;
            }

            extern "C" aclError aclrtSetDevice(int) { return ACL_SUCCESS; }
            extern "C" aclError aclrtMalloc(void**, size_t, int) { return ACL_SUCCESS; }
            extern "C" aclError aclrtMemcpy(void*, size_t, const void*, size_t, int) { return ACL_SUCCESS; }
            extern "C" aclError aclrtFree(void*) { return ACL_SUCCESS; }

            extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t stream)
            {
                const int index = g_launchCount++;
                g_streams[index] = stream;
                std::memcpy(&g_tasks[index], taskInfo, sizeof(g_tasks[index]));
                return g_failOnCall == g_launchCount ? 507000 : RT_ERROR_NONE;
            }

            TileXRDirectCcuTaskInfo MakeTask(uint8_t mission, uint16_t instStart)
            {
                TileXRDirectCcuTaskInfo task {};
                task.dieId = 2;
                task.missionId = mission;
                task.timeout = static_cast<uint16_t>(30 + mission);
                task.instStartId = instStart;
                task.instCnt = 2;
                task.key = 0xabc00000U + mission;
                task.argSize = 13;
                for (uint32_t i = 0; i < 13; ++i) {
                    task.args[i] = 0x8000000000000000ULL + (static_cast<uint64_t>(mission) << 8U) + i;
                }
                return task;
            }

            bool SameTask(const TileXRDirectCcuTaskInfo& expected, const rtCcuTaskInfo_t& actual)
            {
                if (expected.dieId != actual.dieId || expected.missionId != actual.missionId ||
                    expected.timeout != actual.timeout || expected.instStartId != actual.instStartId ||
                    expected.instCnt != actual.instCnt || expected.key != actual.key ||
                    expected.argSize != actual.argSize) {
                    return false;
                }
                for (uint32_t i = 0; i < 13; ++i) {
                    if (expected.args[i] != actual.args[i]) {
                        return false;
                    }
                }
                return true;
            }

            int main()
            {
                TileXRDirectCcuTaskInfo tasks[2] = {MakeTask(7, 101), MakeTask(8, 103)};
                TileXRDirectCcuPreparedTasksPtr prepared = TileXRDirectCcuCreatePreparedForTest(tasks, 2);
                if (prepared == nullptr) {
                    std::printf("missing prepared handle\n");
                    return 1;
                }

                TileXRDirectCcuTaskInfo preview {};
                if (TileXRDirectCcuGetPreparedTask(prepared, 1, &preview) != TileXR::TILEXR_SUCCESS ||
                    preview.missionId != tasks[1].missionId || preview.args[12] != tasks[1].args[12]) {
                    std::printf("prepared task preview mismatch\n");
                    return 2;
                }

                void* stream = reinterpret_cast<void*>(0x12345678ULL);
                TileXRDirectCcuSubmitReport report {};
                const int ret = TileXRDirectCcuSubmitPrepared(prepared, stream, &report);
                if (ret != TileXR::TILEXR_SUCCESS || !report.submitted ||
                    report.taskCount != 2 || report.submittedTaskCount != 2) {
                    std::printf("submit report mismatch ret=%d submitted=%d taskCount=%u submittedTaskCount=%u\n",
                        ret, report.submitted ? 1 : 0, report.taskCount, report.submittedTaskCount);
                    return 3;
                }
                if (g_launchCount != 2 || g_streams[0] != stream || g_streams[1] != stream) {
                    std::printf("launch count or stream mismatch count=%d\n", g_launchCount);
                    return 4;
                }
                if (!SameTask(tasks[0], g_tasks[0]) || !SameTask(tasks[1], g_tasks[1])) {
                    std::printf("runtime task payload mismatch\n");
                    return 5;
                }

                const int destroyRet = TileXRDirectCcuDestroyPrepared(prepared);
                return destroyRet == TileXR::TILEXR_SUCCESS ? 0 : 6;
            }
            """
        )

        result = self.compile_and_run_public_prepared_handle(code)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_public_prepared_handle_reports_null_stream_and_mid_batch_failure(self):
        code = textwrap.dedent(
            r"""
            #include "tilexr_api.h"
            #include "tilexr_types.h"

            #include <cstdint>
            #include <cstdio>
            #include <cstring>

            #include <acl/acl_rt.h>
            #include <runtime/kernel.h>

            extern "C" TileXRDirectCcuPreparedTasksPtr TileXRDirectCcuCreatePreparedForTest(
                const TileXRDirectCcuTaskInfo* tasks, uint32_t taskCount);

            namespace {
            int g_launchCount = 0;
            int g_failOnCall = 0;
            }

            extern "C" aclError aclrtSetDevice(int) { return ACL_SUCCESS; }
            extern "C" aclError aclrtMalloc(void**, size_t, int) { return ACL_SUCCESS; }
            extern "C" aclError aclrtMemcpy(void*, size_t, const void*, size_t, int) { return ACL_SUCCESS; }
            extern "C" aclError aclrtFree(void*) { return ACL_SUCCESS; }

            extern "C" rtError_t rtCCULaunch(rtCcuTaskInfo_t*, rtStream_t)
            {
                ++g_launchCount;
                return g_failOnCall == g_launchCount ? 507000 : RT_ERROR_NONE;
            }

            TileXRDirectCcuTaskInfo MakeTask(uint8_t mission, uint16_t instStart)
            {
                TileXRDirectCcuTaskInfo task {};
                task.dieId = 1;
                task.missionId = mission;
                task.timeout = static_cast<uint16_t>(60 + mission);
                task.instStartId = instStart;
                task.instCnt = static_cast<uint16_t>(4 + mission);
                task.key = 0x12340000U + mission;
                task.argSize = 13;
                for (uint32_t i = 0; i < 13; ++i) {
                    task.args[i] = 0xfeed000000000000ULL + (static_cast<uint64_t>(mission) << 8U) + i;
                }
                return task;
            }

            bool Contains(const char* text, const char* needle)
            {
                return text != nullptr && std::strstr(text, needle) != nullptr;
            }

            int main()
            {
                TileXRDirectCcuTaskInfo tasks[2] = {MakeTask(2, 11), MakeTask(3, 12)};
                TileXRDirectCcuPreparedTasksPtr prepared = TileXRDirectCcuCreatePreparedForTest(tasks, 2);
                if (prepared == nullptr) {
                    return 1;
                }

                TileXRDirectCcuSubmitReport nullStreamReport {};
                int ret = TileXRDirectCcuSubmitPrepared(prepared, nullptr, &nullStreamReport);
                if (ret != TileXR::TILEXR_ERROR_PARA_CHECK_FAIL ||
                    nullStreamReport.submitted || nullStreamReport.taskCount != 2 ||
                    nullStreamReport.submittedTaskCount != 0 ||
                    !Contains(nullStreamReport.message, "missing runtime stream") ||
                    g_launchCount != 0) {
                    std::printf("bad null-stream report ret=%d launchCount=%d message=%s\n",
                        ret, g_launchCount, nullStreamReport.message);
                    return 2;
                }

                g_failOnCall = 2;
                TileXRDirectCcuSubmitReport failReport {};
                ret = TileXRDirectCcuSubmitPrepared(prepared, reinterpret_cast<void*>(0x1ULL), &failReport);
                if (ret != TileXR::TILEXR_ERROR_MKIRT || failReport.submitted ||
                    failReport.taskCount != 2 || failReport.submittedTaskCount != 1 ||
                    !Contains(failReport.message, "task=1") ||
                    !Contains(failReport.message, "rtRet=507000") ||
                    !Contains(failReport.message, "dieId=1") ||
                    !Contains(failReport.message, "missionId=3") ||
                    !Contains(failReport.message, "timeout=63") ||
                    !Contains(failReport.message, "instStartId=12") ||
                    !Contains(failReport.message, "instCnt=7") ||
                    !Contains(failReport.message, "key=0x12340003") ||
                    !Contains(failReport.message, "argSize=13") ||
                    !Contains(failReport.message, "args[0]=0xfeed000000000300") ||
                    !Contains(failReport.message, "args[12]=0xfeed00000000030c") ||
                    g_launchCount != 2) {
                    std::printf("bad mid-batch report ret=%d launchCount=%d taskCount=%u submitted=%u message=%s\n",
                        ret, g_launchCount, failReport.taskCount, failReport.submittedTaskCount,
                        failReport.message);
                    return 3;
                }

                const int destroyRet = TileXRDirectCcuDestroyPrepared(prepared);
                return destroyRet == TileXR::TILEXR_SUCCESS ? 0 : 4;
            }
            """
        )

        result = self.compile_and_run_public_prepared_handle(code)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_public_prepared_handle_test_seam_is_test_only_and_keeps_private_ccu_out(self):
        source = COMM_WRAP_SOURCE.read_text(encoding="utf-8")

        self.assertIn("TILEXR_CCU_TESTING", source)
        self.assertIn("TileXRDirectCcuCreatePreparedForTest", source)
        for needle in [
            "libhcomm",
            "libhccl_v2",
            "libhccl_fwk",
            "libmc2_client",
            "HcclGetCcuTaskInfo",
            "HcomGetCcuTaskInfo",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
