#ifndef TILEXR_COLLECTIVES_PERF_H
#define TILEXR_COLLECTIVES_PERF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void *TileXRCollectivePerfSession;

typedef struct TileXRCollectivePerfConfig {
    int enabled;
    const char *outputDir;
    unsigned int sampleEveryN;
    int emitAiPrompt;
    const char *aiCommand;
} TileXRCollectivePerfConfig;

int TileXRCollectivePerfSessionCreate(const TileXRCollectivePerfConfig *config,
                                      TileXRCollectivePerfSession *session);
int TileXRCollectivePerfSessionDestroy(TileXRCollectivePerfSession session);
int TileXRCollectivePerfSetActiveSession(TileXRCollectivePerfSession session);
int TileXRCollectivePerfWriteReport(TileXRCollectivePerfSession session);
int TileXRCollectivePerfWriteIncompleteReport(TileXRCollectivePerfSession session, const char *reason);

#ifdef __cplusplus
}
#endif

#endif // TILEXR_COLLECTIVES_PERF_H
