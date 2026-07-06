# EgoTraj-Bench L1 Intermediate Data

This guide explains the released `L1-intermediate/` data on Hugging Face and
how to use it for analysis or new data-processing variants.

Dataset link:

```text
https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/L1-intermediate
```

L1 is the bridge between the TBD raw data and the final L2 `.npz` tensors. It
contains clean BEV pedestrian ground truth, BEV GT projected into FPV for
visibility analysis, YOLOv8 + BoTSORT FPV detections/tracks projected back to
BEV/world coordinates, and robot paths for segment windows.

## Structure

```text
L1-intermediate/
├── README.md
├── CHECKSUMS.sha256
├── scene_splits.json
├── bev_gt/
│   ├── segments/
│   └── projected_visibility/
├── fpv_detections/
│   ├── segments/
│   └── merged/
└── robot_paths/
```

## Contents

| Folder | Count | Description |
|--------|------:|-------------|
| `bev_gt/segments/` | 174 | Clean BEV pedestrian GT segments at 2.5 fps |
| `bev_gt/projected_visibility/` | 174 | BEV GT projected into FPV with visibility metadata |
| `fpv_detections/segments/` | 173 | Segment-level FPV detections/tracks projected to BEV/world coordinates |
| `fpv_detections/merged/` | 17 | Scene-level merged FPV detection/tracking outputs |
| `robot_paths/` | 174 | Ego/robot paths for segment windows |
| `scene_splits.json` | 1 | Train/val/test scene split metadata |

## Column Schemas

`bev_gt/segments/*.csv`:

```text
svo_name_core,timestamp,svo_frame_id,scaled_rgb_frame_id,ppl_data_idx,
rgb_frame_id,frame_id,agent_id,x_w,y_w,th,valid,max_svo_frame,min_svo_frame
```

`bev_gt/projected_visibility/*.csv` adds:

```text
svo_time_start,time_passed_in_ms,px_count,from_bev_projected_bbox,max_iou
```

`fpv_detections/segments/*.txt` and `fpv_detections/merged/*.csv`:

```text
svo_name_core,timestamp,svo_frame_id,bbox,confidence,scaled_rgb_frame_id,
ppl_data_idx,rgb_frame_id,x_w_est,y_w_est,valid,max_svo_frame,min_svo_frame,agent_id
```

`robot_paths/<scene>/*.csv`:

```text
frame_id,x,y,det_ids
```

In `robot_paths`, `frame_id` is the sampled SVO frame and `x,y` are the
ego/robot BEV coordinates. Despite the name, `det_ids` is not the FPV tracker
ID list. It is a serialized list of BEV GT pedestrian candidates at the same
frame after projecting BEV GT into the FPV image. Each inner entry has:

```text
[agent_id, x_w, y_w, bbox_left, bbox_top, bbox_width, bbox_height, px_count, max_iou]
```

Here `agent_id,x_w,y_w` come from clean BEV GT, `bbox_*` is the projected FPV
box, and `px_count,max_iou` are the visibility/filtering metadata. `[]` means no
BEV GT pedestrian candidate was associated with that robot frame in this
projection table.

## Application Tracks

### Trajectory Prediction

Use this track if you want to train/evaluate forecasting models or inspect how
the final L2 samples are formed.

Recommended starting point:

- Use `L2-processed/EgoTraj-TBD/*.npz` for direct model training/evaluation.
- Use L1 when you need to inspect the noisy FPV observations, clean BEV GT, or
  scene/segment-level construction.

Relevant L1 files:

- `bev_gt/segments/`: clean BEV pedestrian trajectories (`x_w,y_w`).
- `fpv_detections/segments/`: noisy FPV-derived observed trajectories
  (`x_w_est,y_w_est`) at the segment level.
- `fpv_detections/merged/`: scene-level noisy FPV-derived tracks.
- `robot_paths/`: ego motion and segment windows.
- `scene_splits.json`: released train/val/test scene split metadata.

Note: the released L1 core does not include standalone Hungarian assignment
records. The final matched samples are materialized in L2 as `all_obs` and
`all_pred`.

### Detection / Tracking Analysis

Use this track if you want to study detector/tracker behavior before trajectory
prediction.

Relevant files:

- `fpv_detections/segments/`: segment-level detector/tracker output.
- `fpv_detections/merged/`: scene-level merged tracks.

Key fields:

- `bbox`: FPV image-space pedestrian box.
- `confidence`: detector score.
- `agent_id`: tracker identity.
- `x_w_est,y_w_est`: bbox-derived BEV/world position estimate.
- `valid`: whether the projected observation is considered valid.

These files are useful for studying ID switches, tracking fragmentation,
projection noise, detection confidence, and noisy observation gaps.

### Visibility / Occlusion Analysis

Use this track if you want to understand which BEV GT pedestrians are visible
from the ego view before they are paired with FPV detections.

Relevant files:

- `bev_gt/projected_visibility/`

Key fields:

- `x_w,y_w`: clean BEV GT position.
- `from_bev_projected_bbox`: projected FPV bbox from BEV GT.
- `px_count`: visible pixel count after projection/segmentation filtering.
- `max_iou`: overlap with detected/segmented regions.

This is the most useful L1 component for visibility filtering, occlusion
analysis, and building alternative GT-to-FPV candidate selection rules.

### BEV-FPV Alignment / Raw Frame Lookup

Use this track if you want to map L1 rows back to raw TBD frames or extract
additional visual features.

Relevant files:

- `svo_name_core`, `svo_frame_id`, `timestamp`, and `rgb_frame_id` in BEV/FPV
  files.
- `robot_paths/` for ego path and segment window reconstruction.
- `L0-raw/README.md` for the official TBD raw data entry point.

These fields allow you to connect L1 rows to raw RGB/depth/video frames from
the original TBD data.

## Known Missing Segment

`fpv_detections/segments/` has 173 files while BEV segment data has 174 files.
The missing FPV file is:

```text
fpv-2022-12-07-13-46-01_st-26750_ed-26950.txt
```

This is a known null FPV segment, not a packaging error.

## Integrity Check

After downloading L1, verify file integrity with:

```bash
cd L1-intermediate
sha256sum -c CHECKSUMS.sha256
```
