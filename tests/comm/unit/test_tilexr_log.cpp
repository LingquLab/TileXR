#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "tilexr_log.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

class CaptureStdStreams {
public:
    CaptureStdStreams()
        : oldOut_(std::cout.rdbuf(out_.rdbuf())), oldErr_(std::cerr.rdbuf(err_.rdbuf()))
    {
    }

    ~CaptureStdStreams()
    {
        std::cout.rdbuf(oldOut_);
        std::cerr.rdbuf(oldErr_);
    }

    std::string Output() const
    {
        return out_.str() + err_.str();
    }

private:
    std::ostringstream out_;
    std::ostringstream err_;
    std::streambuf* oldOut_;
    std::streambuf* oldErr_;
};

void ClearLogEnv()
{
    unsetenv("TILEXR_LOG_LEVEL");
    unsetenv("ASDOPS_LOG_LEVEL");
}

std::string ExpensiveMessage(int& calls)
{
    ++calls;
    return "expensive-message";
}

void TestDefaultLevelFiltersDebugAndKeepsInfo()
{
    ClearLogEnv();

    CaptureStdStreams capture;
    TILEXR_LOG(DEBUG) << "debug-hidden";
    TILEXR_LOG(INFO) << "info-visible";

    const auto output = capture.Output();
    CHECK_TRUE(output.find("debug-hidden") == std::string::npos);
    CHECK_TRUE(output.find("info-visible") != std::string::npos);
    CHECK_TRUE(output.find("INFO") != std::string::npos);
}

void TestInvalidLevelFallsBackToInfo()
{
    setenv("TILEXR_LOG_LEVEL", "NOT_A_LEVEL", 1);
    unsetenv("ASDOPS_LOG_LEVEL");

    CaptureStdStreams capture;
    TILEXR_LOG(DEBUG) << "debug-hidden";
    TILEXR_LOG(INFO) << "info-visible";

    const auto output = capture.Output();
    CHECK_TRUE(output.find("debug-hidden") == std::string::npos);
    CHECK_TRUE(output.find("info-visible") != std::string::npos);
}

void TestLogIfDoesNotEvaluateMessageWhenConditionIsFalse()
{
    setenv("TILEXR_LOG_LEVEL", "TRACE", 1);
    unsetenv("ASDOPS_LOG_LEVEL");

    int messageCalls = 0;
    CaptureStdStreams capture;
    TILEXR_LOG_IF(false, INFO) << ExpensiveMessage(messageCalls);

    CHECK_EQ(messageCalls, 0);
    CHECK_TRUE(capture.Output().empty());
}

void TestLogMacroDoesNotStealElse()
{
    ClearLogEnv();

    bool elseBranchRan = false;
    if (false)
        TILEXR_LOG(INFO) << "should-not-print";
    else
        elseBranchRan = true;

    CHECK_TRUE(elseBranchRan);
}

} // namespace

int main()
{
    TestDefaultLevelFiltersDebugAndKeepsInfo();
    TestInvalidLevelFallsBackToInfo();
    TestLogIfDoesNotEvaluateMessageWhenConditionIsFalse();
    TestLogMacroDoesNotStealElse();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR log checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR log checks passed" << std::endl;
    return 0;
}
