#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string& path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string& path)
{
    std::ifstream input(RepoPath(path).c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << RepoPath(path) << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "unexpected text in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

std::string RunCommand(const std::string& command)
{
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        std::cerr << "failed to run command: " << command << std::endl;
        ++g_failures;
        return {};
    }

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    const int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "command failed: " << command << std::endl;
        ++g_failures;
    }
    return output;
}

void CheckNoLineContains(const std::string& description, const std::string& text, const std::string& needle)
{
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(needle) != std::string::npos) {
            std::cerr << "unexpected " << description << ": " << line << std::endl;
            ++g_failures;
        }
    }
}

void TestOpenSourceTarballsAreNotTracked()
{
    const std::string command = "git -C " + RepoPath(".") + " ls-files 3rdparty/open_source";
    const auto trackedFiles = RunCommand(command);

    CheckNoLineContains("tracked open-source dependency archive", trackedFiles, ".tar.gz");
    CheckNoLineContains("tracked open-source dependency archive", trackedFiles, ".tar.xz");
}

void TestCommInitChecksDeviceCommArgsSync()
{
    const std::string path = "src/comm/tilexr_comm.cpp";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "InitMem();\n    g_localPeerMemMap");
    CheckNotContains(path, text, "    SyncCommArgs();");
    CheckContains(path, text, "ret = SyncCommArgs();");
    CheckContains(path, text, "ret = InitMem();");
}

void TestCWrappersDoNotPublishFailedCommunicators()
{
    const std::string path = "src/comm/comm_wrap.cpp";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "*comm = c;\n    int ret = c->Init();");
    CheckNotContains(path, text, "*comm = c;\n    int ret = c->InitThread");
    CheckNotContains(path, text, "comms[i] = new (std::nothrow) TileXRComm");
    CheckContains(path, text, "c.release()");
    CheckContains(path, text, "commHolders[i].release()");
}

void TestDumpInitCleansFailedAllocations()
{
    const std::string path = "src/comm/tilexr_comm.cpp";
    const auto text = ReadFile(path);

    CheckContains(path, text, "aclrtFree(dumpAddr);");
    CheckContains(path, text, "std::free(memory);");
}

void TestSocketExchangeUsesDirectConnectionsOnly()
{
    const std::string cppPath = "src/comm/tools/socket/tilexr_sock_exchange.cpp";
    const std::string headerPath = "src/comm/tools/socket/tilexr_sock_exchange.h";
    const auto cppText = ReadFile(cppPath);
    const auto headerText = ReadFile(headerPath);

    CheckNotContains(cppPath, cppText, "StartSecureTunnel");
    CheckNotContains(headerPath, headerText, "StartSecureTunnel");
    CheckNotContains(cppPath, cppText, "popen");
    CheckNotContains(cppPath, cppText, "/usr/bin/ssh");
    CheckNotContains(headerPath, headerText, "FILE* pipe_");
    CheckNotContains(headerPath, headerText, "lockFileDescriptor_");
    CheckContains(cppPath, cppText, "return Connect();");
    CheckContains(cppPath, cppText, "bool envProvided = false;");
    CheckContains(cppPath, cppText, "if (!envProvided) {");
    CheckNotContains(cppPath, cppText,
                     "handle->addr.sin.sin_port = htons(TILEXR_DEFAULT_SOCK_PORT + dev + commDomain);");
}

void TestRuntimeEnvDoesNotPrependCannDevlib()
{
    const std::string path = "scripts/common_env.sh";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "${ASCEND_HOME_PATH}/${TILEXR_OS_ARCH}-linux/devlib");
}

void TestRuntimeEnvUsesReadableDriverShim()
{
    const std::string path = "scripts/common_env.sh";
    const auto text = ReadFile(path);

    CheckContains(path, text, "TILEXR_DRIVER_SHIM_HOME");
    CheckContains(path, text, "[ ! -r \"${ASCEND_DRIVER_PATH}/kernel/inc\" ]");
    CheckContains(path, text, "${ASCEND_HOME_PATH}/${TILEXR_OS_ARCH}-linux/include/driver");
    CheckContains(path, text, "export ASCEND_DRIVER_PATH=${TILEXR_DRIVER_SHIM_HOME}");
}

void TestRootCMakeRespectsAscendDriverOverride()
{
    const std::string path = "CMakeLists.txt";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)");
    CheckContains(path, text, "set(_tilexr_default_ascend_driver_path \"$ENV{ASCEND_DRIVER_PATH}\")");
    CheckContains(path, text, "set(ASCEND_DRIVER_PATH \"${_tilexr_default_ascend_driver_path}\" CACHE PATH");
}

void TestCommBuildIncludesProfilingHeaders()
{
    const std::string rootPath = "CMakeLists.txt";
    const std::string commPath = "src/comm/CMakeLists.txt";
    const auto rootText = ReadFile(rootPath);
    const auto commText = ReadFile(commPath);

    CheckContains(rootPath, rootText, "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/profiling/");
    CheckContains(commPath, commText, "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/profiling/");
}

void TestChipMapCoversObservedAscend950Variants()
{
    const std::string path = "src/comm/tilexr_internal.cpp";
    const auto text = ReadFile(path);

    CheckContains(path, text, "{\"Ascend950PR\", ChipName::CHIP_950PR}");
    CheckContains(path, text, "{\"Ascend950PR_9589\", ChipName::CHIP_950PR}");
    CheckContains(path, text, "{\"Ascend950PR_9599\", ChipName::CHIP_950PR}");
}

} // namespace

int main()
{
    TestOpenSourceTarballsAreNotTracked();
    TestCommInitChecksDeviceCommArgsSync();
    TestCWrappersDoNotPublishFailedCommunicators();
    TestDumpInitCleansFailedAllocations();
    TestSocketExchangeUsesDirectConnectionsOnly();
    TestRuntimeEnvDoesNotPrependCannDevlib();
    TestRootCMakeRespectsAscendDriverOverride();
    TestCommBuildIncludesProfilingHeaders();
    TestChipMapCoversObservedAscend950Variants();

    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR source guard checks passed" << std::endl;
    return 0;
}
