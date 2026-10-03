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

### Browsing and rebuilding

Collections, their photos, and the library are displayed **newest first**. Collection recency uses the latest photo date, not the collection's machine score or the time it was rebuilt. The default timeline has horizontal year/month separators; **Group by** switches to yearly sections or an ungrouped grid. Filters and pagination preserve that choice, and unknown dates appear last. Date ranges remain visible on collection cards, including sets spanning several years.

**Best of {year}** is a technical-quality shortlist, not an aesthetic model: sharpness, exposure, contrast, color, resolution, and any star ratings influence selection. Exact duplicates and similar burst frames are reduced, with at most five picks per day and up to 60 per year. Photos are then displayed newest first; the **Quality #** badge keeps their independent selection rank. **Keep best** selects by that rank, not by the first photos displayed.

Use **Collections → Rebuild suggestions** or **Activity → Curate only** to regenerate curation from stored analysis. This does not scan the source or download photos. It preserves keep/reject decisions, notes, ratings, workflow status, and renamed titles. Old automatic suggestions that no longer qualify are removed if unreviewed; reviewed or renamed sets are retained.

Trip suggestions now require evidence of departure **and return**: recurring GPS observations around the same base before and after a bounded period away. Dense photo sequences alone, long stays, and moves are not sufficient. The base is inferred from the surrounding time window, allowing it to change when you move. Sparse GPS or unreliable dates produce fewer trip suggestions rather than speculative ones.

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

3. Pull the published image and start the service:

   ```bash
   docker compose pull
   docker compose up -d
   ```

4. Open <http://localhost:8787>.

The default image is [`raregoat8804/photo-book-curator:latest`](https://hub.docker.com/r/raregoat8804/photo-book-curator). Set `IMAGE` to use a pinned version or another registry.

With `AUTO_START=true`, the first full scan starts automatically. It is safe to close the browser. Progress is durable and interrupted work resumes after a normal container restart.

### Portainer

Create a **Stack** from this repository's `docker-compose.yml`, define the variables shown in `.env.example` in Portainer's environment-variable UI, and deploy it. Portainer pulls the published Docker Hub image; it does not need to build the source. The Compose file does not require a repository `.env` file, so Git-based stacks work without committing secrets. Keep the named `photo-curator-data` volume when updating the stack. The default published port is `8787`; set `WEB_PORT` to change it.

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
| `IMAGE` | `raregoat8804/photo-book-curator:latest` | Published image or a pinned alternative such as `:0.1.0` |
| `WEB_PORT` | `8787` | Host port for the web interface |
| `MCP_ENABLED` | `true` | Expose bounded read-only diagnostics at `/mcp/` |
| `MCP_BEARER_TOKEN` | empty | Optional bearer token; recommended on shared networks |
| `MCP_ALLOWED_HOSTS` | empty | Optional exact Host-header allowlist for DNS-rebinding protection |
| `MCP_LOG_ENTRIES` | `200` | Size of the sanitized in-memory warning/error ring |
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
| `TRIP_HOME_RADIUS_KM` | `30` | Radius of a recurring base and a return observation |
| `TRIP_MIN_DISTANCE_KM` | `100` | Travel threshold; always at least twice the base radius |
| `TRIP_MAX_DAYS` | `30` | Maximum departure-to-return interval for a trip suggestion |
| `TRIP_CONTEXT_DAYS` | `90` | Surrounding GPS-day evidence window for the time-local base |

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

## Read-only MCP diagnostics

A Streamable HTTP MCP endpoint is available at:

```text
http://SERVER:8787/mcp/
```

It is designed for low-token operational debugging from Pi. Its eight tools provide a compact deployment overview, a read-only source connectivity check, durable job history, paginated photo issues, one-photo diagnostics, collection summaries, a bounded consistency audit, and a sanitized in-memory warning/error ring.

The MCP surface is intentionally narrower than the web application:

- every tool declares the MCP read-only annotation;
- there is no arbitrary SQL, filesystem, source-photo, preview, or log-file access;
- no tool can enqueue/cancel jobs or change ratings, decisions, or collections;
- credentials and deployment identifiers are removed from errors and diagnostic logs;
- list results are bounded and cursor-paginated;
- photo paths are excluded by default and must be explicitly requested.

For a shared LAN, generate a long random value for `MCP_BEARER_TOKEN` in Portainer. Keep the same value in a local environment variable used by Pi—do not put it directly in `mcp.json`:

```bash
export PHOTO_CURATOR_MCP_TOKEN='the value configured in Portainer'
pi mcp add photo-curator \
  --url http://SERVER:8787/mcp/ \
  --bearer-token-env-var PHOTO_CURATOR_MCP_TOKEN \
  --exposure codemode
pi mcp list
```

If the app and Pi run on the same host, use `http://127.0.0.1:8787/mcp/`. If no bearer token is configured, omit `--bearer-token-env-var`; only do this on a trusted private network. Run `/reload` in an existing Pi session after adding or changing the server.

`MCP_ALLOWED_HOSTS` optionally enables exact Host-header validation. Values are comma-separated and must include the port or the SDK's `:*` port wildcard, for example `photos.local:8787,photos.local:*,192.168.1.20:*`. Leave it empty when connecting through changing LAN addresses and rely on network isolation plus the bearer token.

## Development

To build and run the container from the current checkout:

```bash
docker build -t photo-book-curator:local .
IMAGE=photo-book-curator:local docker compose up -d
```

For native development:

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
