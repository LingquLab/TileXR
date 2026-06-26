# Open-Source Dependency Archives

This directory is populated on demand by:

```bash
bash scripts/download_open_source_deps.sh
```

The downloaded archives are intentionally ignored by Git to keep repository
history small. `scripts/prepare.sh` runs the downloader before it builds local
utility dependencies.
