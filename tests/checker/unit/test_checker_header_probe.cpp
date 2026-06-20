#include "tilexr/checker/case.h"
#include "tilexr/checker/executor.h"
#include "tilexr/checker/shim_runtime.h"
#include "kernel_operator.h"
#include "collective_utils.h"

int main() {
    tilexr::checker::CheckerCase test_case{};
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    return 0;
}
