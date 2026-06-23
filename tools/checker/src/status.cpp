#include "tilexr/checker/status.h"

namespace tilexr {
namespace checker {

CheckerStatus CheckerStatus::Ok() {
    return CheckerStatus();
}

CheckerStatus CheckerStatus::Fail(const std::string &message) {
    CheckerStatus status;
    status.code = CheckerStatusCode::kFail;
    status.message = message;
    return status;
}

CheckerStatus CheckerStatus::Unsupported(const std::string &message) {
    CheckerStatus status;
    status.code = CheckerStatusCode::kUnsupported;
    status.message = message;
    return status;
}

bool CheckerStatus::ok() const {
    return code == CheckerStatusCode::kOk;
}

}  // namespace checker
}  // namespace tilexr
