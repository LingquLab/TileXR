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
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << fullPath << std::endl;
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
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckDoesNotContain(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestHeaderShape()
{
    const std::string path = "src/include/tilexr_data_as_flag.h";
    const auto text = ReadFile(path);
    CheckContains(path, text, "DATA_AS_FLAG_BLOCK_BYTES = 512");
    CheckContains(path, text, "DATA_AS_FLAG_PAYLOAD_BYTES = 480");
    CheckContains(path, text, "DATA_AS_FLAG_FLAG_BYTES = 32");
    CheckContains(path, text, "DATA_AS_FLAG_FLAG_OFFSET_BYTES = 480");
    CheckContains(path, text, "DATA_AS_FLAG_FLAG_FLOATS");
    CheckContains(path, text, "DATA_AS_FLAG_READY_VALUE = 1.0f");
    CheckContains(path, text, "DataAsFlagBlockCountForPayloadBytes");
    CheckContains(path, text, "DataAsFlagInit");
    CheckContains(path, text, "DataAsFlagSend");
    CheckContains(path, text, "DataAsFlagCheck");
    CheckContains(path, text, "DataAsFlagCheckAndRecv");
}

void TestHeaderUsesExpectedAscendCApis()
{
    const std::string path = "src/include/tilexr_data_as_flag.h";
    const auto text = ReadFile(path);
    CheckContains(path, text, "#include \"adv_api/reduce/sum.h\"");
    CheckContains(path, text, "AscendC::DataCopyPad");
    CheckContains(path, text, "AscendC::Sum<");
    CheckContains(path, text, "AscendC::SumParams");
    CheckContains(path, text, "sharedTmpBuffer");
    CheckContains(path, text, "DataAsFlagSumWorkspaceBytes");
    CheckContains(path, text, "ReinterpretCast<float>");
    CheckContains(path, text, "GetSize()");
    CheckContains(path, text, "DATA_AS_FLAG_SUM_RESULT_BYTES");
    CheckContains(path, text, "DataAsFlagMaxRecvBlocks");
    CheckContains(path, text, "while (!DataAsFlagCheckBatch");
    CheckDoesNotContain(path, text, "GlobalTensor::GetValue");
    CheckDoesNotContain(path, text, "GlobalTensor::SetValue");
    CheckDoesNotContain(path, text, "checkScratchBlockCapacity");
}

void TestInstallWiring()
{
    const std::string path = "src/comm/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_data_as_flag.h");
    CheckContains(path, text, "DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}");
}

void TestRootCMakeWiring()
{
    const std::string path = "CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "add_subdirectory(tests/data_as_flag)");
}

void TestDataAsFlagCMakeWiring()
{
    const std::string path = "tests/data_as_flag/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "add_executable(test_tilexr_data_as_flag_header_compile");
    CheckContains(path, text, "unit/test_tilexr_data_as_flag_header_compile.cpp");
    CheckContains(path, text, "add_executable(test_tilexr_data_as_flag_source_guard");
    CheckContains(path, text, "unit/test_tilexr_data_as_flag_source_guard.cpp");
    CheckContains(path, text, "target_include_directories(test_tilexr_data_as_flag_header_compile PRIVATE");
    CheckContains(path, text, "target_compile_definitions(test_tilexr_data_as_flag_source_guard PRIVATE");
    CheckContains(path, text, "add_test(NAME test_tilexr_data_as_flag_header_compile");
    CheckContains(path, text, "add_test(NAME test_tilexr_data_as_flag_source_guard");
    CheckContains(path, text, "install(TARGETS");
    CheckContains(path, text, "test_tilexr_data_as_flag_header_compile");
    CheckContains(path, text, "test_tilexr_data_as_flag_source_guard");
}

} // namespace

int main()
{
    TestHeaderShape();
    TestHeaderUsesExpectedAscendCApis();
    TestInstallWiring();
    TestRootCMakeWiring();
    TestDataAsFlagCMakeWiring();
    if (g_failures != 0) {
        std::cerr << g_failures << " DataAsFlag source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR DataAsFlag source guard checks passed" << std::endl;
    return 0;
}
