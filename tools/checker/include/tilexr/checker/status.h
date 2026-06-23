#ifndef TILEXR_CHECKER_STATUS_H
#define TILEXR_CHECKER_STATUS_H

#include <string>

namespace tilexr {
namespace checker {

enum class CheckerStatusCode {
    kOk,
    kFail,
    kUnsupported,
    kInconclusive,
    kInternalError
};

struct CheckerStatus {
    CheckerStatusCode code = CheckerStatusCode::kOk;
    std::string message;

    static CheckerStatus Ok();
    static CheckerStatus Fail(const std::string &message);
    static CheckerStatus Unsupported(const std::string &message);
    bool ok() const;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_STATUS_H
