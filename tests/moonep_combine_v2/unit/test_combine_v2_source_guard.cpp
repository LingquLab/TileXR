#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TILEXR_SOURCE_ROOT
#error TILEXR_SOURCE_ROOT must be defined
#endif

namespace {

std::string Read(const std::string &path)
{
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string Lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

bool Require(const std::string &text, const std::string &needle,
    const char *message)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool Reject(const std::string &text, const std::string &needle,
    const char *message)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool RequireBefore(const std::string &text, const std::string &first,
    const std::string &second, const char *message)
{
    const std::size_t firstPos = text.find(first);
    const std::size_t secondPos = text.find(second);
    if (firstPos == std::string::npos || secondPos == std::string::npos ||
        firstPos >= secondPos) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    const std::string root = TILEXR_SOURCE_ROOT;
    const std::string rootCmake = Read(root + "/CMakeLists.txt");
    const std::string moonEpCmake = Read(root + "/src/moonep/CMakeLists.txt");
    const std::string v1Cmake = Read(root + "/src/moonep/combine/CMakeLists.txt");
    const std::string v2Cmake = Read(root + "/src/moonep/combine_v2/CMakeLists.txt");
    const std::string kernelEntry = Read(root +
        "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.cpp");
    const std::string kernelImpl = Read(root +
        "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h");
    const std::string host = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_host.cpp");
    const std::string launch = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_launch.cpp");
    const std::string publicHeader = Read(root +
        "/src/include/tilexr_moonep_combine_v2.h");
    const std::string registration = Read(root +
        "/src/moonep/common/moonep_kernel_registration.h");

    bool ok = true;
    ok &= Require(rootCmake, "add_subdirectory(tests/moonep_combine_v2)",
        "Combine V2 tests are not wired at the root");
    ok &= Require(moonEpCmake, "add_subdirectory(combine_v2)",
        "Combine V2 source directory is not configured");
    ok &= Require(moonEpCmake, "tilexr-moonep-combine-v2",
        "Combine V2 is not linked into the MoonEP aggregate library");
    ok &= Reject(v1Cmake, "combine_v2",
        "Combine V1 build definition contains V2 changes");

    ok &= Require(v2Cmake, "add_library(tilexr-moonep-combine-v2 SHARED",
        "Combine V2 does not own a standalone shared library");
    ok &= Require(v2Cmake, "tilexr_moonep_combine_v2_kernel.cpp",
        "Combine V2 kernel entry is not the build source");
    ok &= Require(v2Cmake, "tilexr_moonep_combine_v2_kernel.h",
        "Combine V2 kernel implementation is not a build dependency");
    ok &= Require(v2Cmake, "-DCATLASS_ARCH=3510",
        "Combine V2 CATLASS target is missing");
    ok &= Require(v2Cmake, "--cce-auto-sync",
        "Combine V2 auto synchronization is missing");
    ok &= Require(v2Cmake, "SOVERSION 2",
        "Combine V2 SONAME is not version 2");
    ok &= Reject(v2Cmake, "/devlib",
        "Combine V2 links the toolkit stub library directory");

    ok &= Require(kernelEntry, "#include \"tilexr_moonep_combine_v2_kernel.h\"",
        "Combine V2 entry does not include its implementation header");
    ok &= Require(kernelEntry, "tilexr_moonep_combine_v2_kernel",
        "Combine V2 operator entry is missing");
    ok &= Require(kernelEntry, "op.Init(",
        "Combine V2 entry does not initialize the implementation");
    ok &= Require(kernelEntry, "op.Process();",
        "Combine V2 entry does not execute the implementation");
    ok &= RequireBefore(kernelEntry,
        "GetBlockIdx()) >= activeCoreCount", "AscendC::TPipe pipe;",
        "Combine V2 inactive blocks do not exit before TPipe construction");
    ok &= Reject(kernelEntry, "TileXR::UDMA",
        "Combine V2 entry contains algorithm implementation");
    ok &= Reject(kernelEntry, "InitBuffers",
        "Combine V2 entry contains implementation helpers");
    ok &= Reject(kernelEntry, "<<<",
        "Combine V2 entry contains Host launch syntax");
    if (static_cast<size_t>(std::count(kernelEntry.begin(),
            kernelEntry.end(), '\n')) > 40U) {
        std::cerr << "Combine V2 kernel entry is no longer thin\n";
        ok = false;
    }

    ok &= Require(kernelImpl, "class MoonEpCombineV2",
        "Combine V2 implementation class is missing from the header");
    ok &= Require(kernelImpl, "TileXR::UDMA",
        "Combine V2 implementation does not use UDMA");
    ok &= Require(kernelImpl, "UDMA_SHARED_QP",
        "Combine V2 implementation does not require shared QPs");
    ok &= Require(kernelImpl, "Simt::VF_CALL<MoonEpCombineV2BuildPayloadVf>",
        "Combine V2 implementation does not use its SIMT WQE builder");
    ok &= Require(kernelImpl, "st_dev(",
        "Combine V2 implementation does not ring device doorbells");
    ok &= Require(kernelImpl, "rank_, 0U, core_, rankSize_",
        "Combine V2 implementation does not use the runtime Ring schedule");
    ok &= Require(kernelImpl, "step < stepCount_",
        "Combine V2 implementation does not use the runtime step count");
    ok &= Require(kernelImpl, "core < activeCoreCount_",
        "Combine V2 failure convergence does not use active cores");
    ok &= Require(kernelImpl, "encoded, slots_, rankSize_",
        "Combine V2 destination validation does not use runtime rank size");
    ok &= Require(kernelImpl, "kMoonEpCombineV2MaxSourcesPerCore",
        "Combine V2 inbound Done polling is not rank generalized");
    ok &= Reject(kernelImpl, "SyncAll<true>();",
        "Combine V2 implementation still uses whole-launch barriers");
    ok &= Reject(kernelImpl, "extern \"C\" __global__",
        "Combine V2 implementation header contains the operator entry");
    ok &= Reject(kernelImpl, "<<<",
        "Combine V2 implementation contains Host launch syntax");

    ok &= Require(host, "UDMA_SHARED_QP",
        "Combine V2 Host does not validate shared-QP capability");
    ok &= Require(host, "TileXRUDMAGetQpCount",
        "Combine V2 Host does not validate the QP count");
    ok &= Require(launch, "LaunchRegisteredMoonEpKernel",
        "Combine V2 does not use registered Runtime launch");
    ok &= Require(launch, "ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE",
        "Combine V2 launch does not query the SIMT UB capacity");
    ok &= Require(launch, "cfgInfo.localMemorySize",
        "Combine V2 launch does not configure SIMT local memory");
    ok &= Require(launch, "params.aivCoreNum",
        "Combine V2 launch does not use the requested AIV count");
    ok &= Require(launch, "kCombineV2KernelSignature",
        "Combine V2 launch signature is missing");
    ok &= Reject(launch, "<<<",
        "Combine V2 Host contains compiler launch syntax");
    ok &= Require(registration, "kCombineV2KernelSignature",
        "Combine V2 stable registration signature is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineGetWorkspaceSizeV2",
        "Combine V2 workspace API is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineV2",
        "Combine V2 launch API is missing");

    const std::vector<std::string> v2Files {
        root + "/src/include/tilexr_moonep_combine_v2.h",
        root + "/src/moonep/combine_v2/CMakeLists.txt",
        root + "/src/moonep/combine_v2/common/combine_v2_profile.h",
        root + "/src/moonep/combine_v2/common/combine_v2_schedule.h",
        root + "/src/moonep/combine_v2/common/combine_v2_wqe_batch.h",
        root + "/src/moonep/combine_v2/host/combine_v2_layout.h",
        root + "/src/moonep/combine_v2/host/combine_v2_layout.cpp",
        root + "/src/moonep/combine_v2/host/combine_v2_host.h",
        root + "/src/moonep/combine_v2/host/combine_v2_host.cpp",
        root + "/src/moonep/combine_v2/host/combine_v2_launch.h",
        root + "/src/moonep/combine_v2/host/combine_v2_launch.cpp",
        root + "/src/moonep/combine_v2/host/tilexr_moonep_combine_v2.cpp",
        root + "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h",
        root + "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.cpp",
    };
    std::string allV2;
    for (const std::string &path : v2Files) {
        allV2 += Lower(Read(path));
    }
    const std::string legacyKeyword = std::string("pair") + "wise";
    ok &= Reject(allV2, legacyKeyword,
        "Combine V2 sources contain the legacy algorithm keyword");
    return ok ? 0 : 1;
}
