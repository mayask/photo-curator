# Photo Book Curator

A self-hosted, local-first photo library analyzer that turns a large read-only archive into reviewable photo-book suggestions. It scans an SMB share (or a read-only local mount), stores only metadata and generated previews locally, and exposes a simple web review workflow.

> **Source safety:** the source abstraction has no write or delete operation. SMB files are opened with `mode="rb"`; local deployments should mount `/photos:ro`. The Compose container has a read-only root filesystem, drops Linux capabilities, and writes only to its `/data` volume (plus an ephemeral `/tmp`).

## What it finds

- trips and time/location-based events
- existing source folders reused as human-authored album clues
- the most photographed and highest-quality days each year
- print-worthy highlights with burst and duplicate suppression
- exact duplicates (SHA-256) and near duplicates (perceptual hashes)
- rapid series/bursts, with the strongest frames ranked first
- recurring calendar dates and optionally named public holidays
- GPS place clusters with offline nearest-city labels
- face clusters, people repeatedly seen together, and short-lived clusters that may represent visitors
- portraits, shared-selfie threads, scenery, and year highlights

Every collection is a suggestion. Originals are never removed or modified. In the web UI you can keep/reject candidates, rate photos, rename suggested collections, change workflow status, and export a CSV manifest for printing or a later copy step.

## How analysis works

Each supported photo is transferred from the NAS at most once per analysis version into a temporary local file. During that pass the worker:

1. calculates a cryptographic content hash;
2. extracts EXIF capture time, camera, orientation, and GPS;
3. creates local thumbnail and review-size JPEG derivatives;
4. measures sharpness, clipping/exposure, contrast, colorfulness, and resolution;
5. calculates perceptual hashes and a compact visual color signature;
6. detects faces with YuNet and creates local SFace embeddings; and
7. checkpoints the result in SQLite before moving to the next image.

Collection generation is a fast database-only pass. It combines adaptive time gaps, GPS distance, density, DBSCAN clusters, face identity embeddings, perceptual distance, exact hashes, technical quality, and diversity constraints. See [`docs/design.md`](docs/design.md) for details and research references.

## Quick start with Docker Compose

1. Copy the sample configuration:

   ```bash
   cp .env.example .env
   ```

2. Set at least these values in `.env`:

   ```dotenv
   SOURCE_MODE=smb
   SMB_HOST=192.168.1.10
   SMB_SHARE=photos
   SMB_PATH=family/library
   SMB_USER=photo-reader
   SMB_PASSWORD=your-password
   ```

3. Start the service:

   ```bash
   docker compose up -d --build
   ```

4. Open <http://localhost:8787>.

With `AUTO_START=true`, the first full scan starts automatically. It is safe to close the browser. Progress is durable and interrupted work resumes after a normal container restart.

### Portainer

Create a **Stack** from this repository's `docker-compose.yml`, define the variables shown in `.env.example` in Portainer's environment-variable UI, and deploy it. The Compose file does not require a repository `.env` file, so Git-based stacks work without committing secrets. Keep the named `photo-curator-data` volume when updating the stack. The default published port is `8787`; set `WEB_PORT` to change it.

Only one application replica/worker is supported with SQLite. Do not scale this service above one container.

## Alternative: a read-only mounted library

If Docker already mounts the NAS (or the photos are local), use:

```dotenv
SOURCE_MODE=local
PHOTO_ROOT=/photos
```

Add a read-only bind or CIFS volume to the service:

```yaml
volumes:
  - photo-curator-data:/data
  - /mnt/nas/photos:/photos:ro
```

The native SMB mode is simpler in Portainer and does not require a privileged container or a host CIFS mount.

## Important configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SOURCE_MODE` | `smb` | `smb` or `local` |
| `SMB_HOST`, `SMB_SHARE`, `SMB_PATH` | | Read-only SMB target |
| `SMB_USER`, `SMB_PASSWORD`, `SMB_DOMAIN` | | SMB credentials/domain |
| `DATA_DIR` | `/data` in Docker | Database, thumbnails, previews, temporary files |
| `AUTO_START` | `true` | Start initial and periodic full scans |
| `SCAN_INTERVAL_HOURS` | `24` | Rescan interval; `0` disables periodic scans |
| `SCAN_MAX_FILES` | `0` | QA/trial cap; `0` scans all files, capped scans disable missing-file cleanup |
| `MAX_FILE_MB` | `250` | Per-file transfer safety limit |
| `MAX_IMAGE_MEGAPIXELS` | `100` | Decoded-image memory safety limit |
| `THUMB_SIZE` / `PREVIEW_SIZE` | `360` / `1600` | Longest edge of local derivatives |
| `FACE_ANALYSIS` | `true` | YuNet detection and SFace local embeddings |
| `HOLIDAY_COUNTRY` | empty | Optional code such as `US`, `DE`, or `GB` |
| `EVENT_GAP_HOURS` | `8` | Maximum gap inside a candidate event |
| `GPS_CLUSTER_KM` | `25` | Broad place-cluster radius |

All supported variables are documented in [`.env.example`](.env.example).

## Long-running behavior

- SQLite runs in WAL mode with a busy timeout.
- Jobs, scan tokens, and per-photo states are persistent.
- Analysis commits after each photo; a restart does not discard completed work.
- SMB directory scans and whole-file reads use bounded retries and connection resets.
- A partially unreadable scan does **not** mark unseen photos as deleted.
- Changed files are reanalyzed based on source size and modification time.
- Collection keys are stable where possible, preserving manual keep/reject decisions across rebuilds.
- `SIGTERM` gets a 30-second Compose grace period; the current file is the largest restart unit.

Back up the `photo-curator-data` volume to preserve analysis and review decisions. The source photos themselves are not part of that volume.

## Supported files and limitations

JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP, and the first frame of GIF images are supported. RAW and video analysis are not included in the first release. Capture time falls back from EXIF to a timestamp in the filename and finally to source modification time.

Face labels are deliberately anonymous (`Person N`) and can make mistakes. GPS city labels are an offline nearest-city approximation. Technical quality scores are ranking aids—not aesthetic truth. Nothing is automatically deleted.

The app intentionally has no login because it is designed for a trusted private network. Do not publish it directly to the internet without an authenticating reverse proxy.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
uvicorn app.main:app --reload
```

For a local development library, set `SOURCE_MODE=local`, `PHOTO_ROOT` to a small test directory, `DATA_DIR=./data`, and `DATABASE_URL=sqlite:///./data/photo-curator.db`.

## Health and API

- `GET /healthz` — container health
- `GET /api/status` — active job and analysis counts
- `POST /api/jobs/full` — scan, analyze, and rebuild collections
- `POST /api/jobs/scan`, `/analyze`, `/collections` — individual phases
- `POST /api/jobs/{id}/cancel` — cooperative cancellation

The web app uses the same API for review decisions and ratings.
