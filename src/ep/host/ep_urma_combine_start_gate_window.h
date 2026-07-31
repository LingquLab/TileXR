#ifndef TILEXR_EP_HOST_EP_URMA_COMBINE_START_GATE_WINDOW_H
#define TILEXR_EP_HOST_EP_URMA_COMBINE_START_GATE_WINDOW_H

namespace TileXREp {

class EpUrmaCombineStartGateWindow {
public:
    bool BeginLaunch()
    {
        const bool runStartGate = firstLaunchAfterSynchronization_;
        firstLaunchAfterSynchronization_ = false;
        return runStartGate;
    }

    void StreamSynchronized()
    {
        firstLaunchAfterSynchronization_ = true;
    }

private:
    bool firstLaunchAfterSynchronization_ = true;
};

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_URMA_COMBINE_START_GATE_WINDOW_H
