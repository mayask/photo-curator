# Design and curation heuristics

## Product principle

The application optimizes for **useful candidates to review**, not autonomous deletion. Personal-photo quality and importance are subjective, so the system keeps alternatives visible and explains why a frame was suggested.

## Pipeline

```text
read-only SMB/local source
        │
        ├─ recursive inventory (size + mtime checkpoints)
        │
        └─ one bounded transfer per changed image
             ├─ SHA-256 exact identity
             ├─ EXIF time/camera/GPS
             ├─ pHash + dHash + HSV signature
             ├─ technical quality metrics
             ├─ local thumbnail/preview
             └─ YuNet faces → SFace embeddings
                         │
                         ▼
                    SQLite (WAL)
                         │
             collection heuristic passes
                         │
                         ▼
               human review + CSV manifest
```

The expensive source-I/O phase is separate from collection generation. New heuristics can therefore be rerun without rereading the NAS.

## Current collection techniques

### Events and trips

Photos are ordered by capture time. A new event starts after a configurable time gap, a meaningful overnight gap, or rapid long-distance movement when both images have GPS. Dense adjacent event-days form trip candidates, bounded to avoid turning a whole year of daily photos into one trip.

Existing non-generic source folders containing 5–1,000 analyzed photos are also retained as album clues. Folder names often encode human knowledge (a trip, wedding, person, or project) that should complement rather than be discarded by automatic clustering.

Time-based grouping follows the established personal-photo organization pattern of using time as the primary cue and location as a refinement. PhotoTOC is an early example of hierarchical event/scene clustering for personal photo browsing:

- Matthew Cooper et al., “Temporal Event Clustering for Digital Photo Collections” / PhotoTOC: <https://www.microsoft.com/en-us/research/publication/phototoc-automatic-clustering-for-browsing-personal-photographs/>
- J. C. Platt et al., “PhotoTOC: Automatic Clustering for Browsing Personal Photographs”: <https://www.microsoft.com/en-us/research/wp-content/uploads/2003/01/phototoc.pdf>

### Duplicate and series handling

- **Exact duplicates:** SHA-256, independent of path and filename.
- **Near duplicates:** library-wide 64-bit perceptual hash lookup using a chunked Hamming-distance index, so resized exports can match even when their timestamps differ.
- **Bursts:** consecutive capture-time gaps below a configurable threshold.
- **Keeper ranking:** quality first, while suppressing repeated content hashes and very close pHashes.

Perceptual hashing is designed to compare content despite small representation changes, unlike cryptographic hashes:

- pHash documentation: <https://phash.org/docs/howto.html>

The app never deletes duplicate candidates because resized exports, edits, and source originals may all be intentionally retained.

### Face grouping and possible visitors

OpenCV Zoo's YuNet detector finds faces and five-point landmarks. SFace aligns each crop and produces a normalized embedding. DBSCAN with cosine distance groups repeated identities without requiring a known number of people; above 5,000 faces, bounded-summary BIRCH clustering avoids DBSCAN's pairwise scaling. The stable cluster identifier is based on the earliest face row rather than a display name. Clusters whose entire appearance is confined to a short period are surfaced as “possible visitor” suggestions. Repeated face co-occurrences also create “people together” threads, which can reveal relationships, visits, and shared selfies.

- OpenCV DNN face detection/recognition tutorial: <https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html>
- OpenCV Zoo models: <https://github.com/opencv/opencv_zoo>

Models run locally on CPU. Face outputs are probabilistic and must be reviewed.

### Places

EXIF GPS coordinates are clustered with DBSCAN using haversine distance. Nearest-city labels come from the offline `reverse_geocoder` dataset, so analysis does not disclose coordinates to an online geocoder. Time remains primary because GPS is often missing and a broad place (such as home) can span many unrelated events.

### Quality and print-worthiness

The lightweight quality score combines:

- Laplacian sharpness (log-normalized);
- midpoint exposure and clipped-pixel penalty;
- luminance contrast;
- colorfulness;
- print-resolution headroom; and
- a small face-presence/prominence bonus.

The score is only used for relative ordering. A diverse-selection pass limits one burst/duplicate family and prevents one heavily photographed day from dominating a year or global highlight set. User ratings contribute to future rankings while remaining separate from machine metrics.

Learned aesthetic scoring (for example NIMA) could be an optional future plugin, but it introduces a large model and aesthetic bias. NIMA's original framing—predicting a distribution of human ratings rather than objective truth—is relevant:

- Talebi and Milanfar, “NIMA: Neural Image Assessment”: <https://arxiv.org/abs/1709.05424>

### Notable and recurring dates

The curator surfaces each year's highest-volume days, repeated month/day combinations across years, and optionally holidays for a configured country. This finds birthdays, anniversaries, annual visits, and traditions without requiring calendar access.

## Inspiration and differentiation

Existing local photo managers validate the usefulness of moments, people, places, and semantic organization:

- PhotoPrism Moments: <https://docs.photoprism.app/user-guide/organize/moments/>
- Immich features: <https://immich.app/features>
- LibrePhotos automatic albums: <https://docs.librephotos.com/docs/user-guide/albums/>

Photo Book Curator is narrower: it leaves the source read-only, prioritizes explainable print shortlists, preserves review decisions, and exports manifests instead of becoming the system of record for originals.

## Robustness properties

1. Scan discovery is committed in batches.
2. Analysis is committed per photo.
3. `.part` files and generated derivatives are atomically renamed.
4. A scan with any unreadable directory skips missing-file deactivation.
5. Running jobs return to `queued` after process restart; stale temporary originals are removed at startup.
6. An attempt is recorded before source I/O, preventing one pathological file from causing an infinite crash/restart loop.
7. SMB directory operations and full-file transfers retry with bounded exponential delay.
8. Source paths are normalized and traversal outside the configured root is rejected.
9. The source interface exposes only walking and binary reads.
10. SQLite uses WAL, foreign keys, and a busy timeout.
11. Human decisions survive heuristic rebuilds even when a photo drops out of the newly generated candidate set.
12. One worker is used to avoid saturating a slow NAS and to keep SQLite ownership simple.

## Future extensions

- optional CLIP/OpenCLIP semantic embeddings and text search;
- learned NIMA/MUSIQ aesthetic scorer as an opt-in model;
- RAW sidecar extraction and video keyframes;
- editable face names and merge/split controls;
- home-location inference to distinguish trips from local events;
- map view and route-aware trip segmentation;
- print-layout/contact-sheet PDF generation;
- export/copy of kept originals to a separate writable destination;
- PostgreSQL and multiple workers for very large libraries.
