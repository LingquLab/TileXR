#include "ep_urma_combine_start_gate_window.h"

#include <iostream>

namespace {

int g_failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void TestOnlyFirstLaunchAfterSynchronizationUsesGate()
{
    TileXREp::EpUrmaCombineStartGateWindow window;
    Check(window.BeginLaunch(), "the first launch must use the start gate");
    Check(!window.BeginLaunch(), "the second launch in a window must skip the start gate");
    Check(!window.BeginLaunch(), "later launches in a window must skip the start gate");

    window.StreamSynchronized();
    Check(window.BeginLaunch(), "the first launch after synchronization must use the start gate");
    Check(!window.BeginLaunch(), "only one launch after synchronization may use the start gate");
}

void TestWindowOneUsesGateForEveryLaunch()
{
    TileXREp::EpUrmaCombineStartGateWindow window;
    for (int launch = 0; launch < 4; ++launch) {
        Check(window.BeginLaunch(), "window=1 must use the start gate for every launch");
        window.StreamSynchronized();
    }
}

} // namespace

int main()
{
    TestOnlyFirstLaunchAfterSynchronizationUsesGate();
    TestWindowOneUsesGateForEveryLaunch();
    return g_failures == 0 ? 0 : 1;
}
