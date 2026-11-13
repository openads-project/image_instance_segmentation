# Configuration and diagnostics

Configure the node with [params.yml](../image_instance_segmentation/config/params.yml).
The [node reference](../image_instance_segmentation/README.md) lists its interfaces
and launch arguments.

## Cameras and outputs

List camera names in `cameras` and configure each under `camera.<name>`:

| Parameter | Meaning |
| --- | --- |
| `camera.<name>.sensor_id` | Unique camera ID for perception observations; defaults to its 1-based position in `cameras` |
| `camera.<name>.image_topic` | Base image topic |
| `camera.<name>.transport` | `raw` or `compressed`; omit `/compressed` from the base topic |
| `camera.<name>.output_prefix` | Prefix for camera output topics |

The supplied parameter file configures one camera with raw transport, enables
tracking, and selects RF-DETR-Seg decoding.

For the default camera, launch arguments remap the input and all output topics:

```bash
ros2 launch image_instance_segmentation image_instance_segmentation.launch.py \
  image_topic:=/camera/image_rect detections_topic:=/perception/detections
```

For multiple cameras, set `cameras: [left, right]` and configure each
`camera.<name>` entry. Their default inputs are `~/<name>/image` and outputs
are under `~/<name>/segmentation/`.

Outputs preserve the input image dimensions, timestamp, and frame ID.
Subscriptions use sensor-data QoS. Class masks contain `model.class_ids`;
zero is background. Instance masks contain per-frame IDs or per-camera track IDs.
`Detection2D.id` contains the track ID when tracking is enabled. Track IDs can be
reused after a track ends.

In the topic tables, `config.image_topic` is the configured input and `joinTopic`
places outputs under the camera's prefix. Parameter names `image_topic`,
`sensor_id`, `transport`, and `output_prefix` are per-camera suffixes.

## Perception objects

Enable each object-list output independently at startup:

```yaml
perception.publish_road_users: true
perception.publish_traffic_lights: true
perception.publish_traffic_signs: true
```

Topics are `perception/road_users`, `perception/traffic_lights` and
`perception/traffic_signs` under each camera's output prefix. They carry
`perception_msgs/msg/ObjectList` with `HEXAMOTION`, `TRAFFICLIGHT` and `TRAFFICSIGN`
states respectively. Metric components remain unmeasured; light state, sign type
and road-user light states remain unknown.

The standard `model.class_names` select the category and road-user classification,
independently of `model.class_ids`. Unmapped names remain in segmentation outputs.
Object IDs persist for a tracked object's lifetime and are not recycled when
mask IDs are reused. Without tracking, each published detection gets a
fresh ID. IDs are scoped to a camera stream and node lifetime.

Observations retain source coordinates and headers. `perception.include_crops`
and `perception.include_masks` independently include BGR crops and binary `mono8`
ROI masks. Pixels are prepared only for subscribed outputs; empty lists are
published for processed frames without matching detections.

## Triton model

Set `prediction.triton_server_url`, `prediction.model_name`, and
`prediction.model_version` for your model, and select its output decoding with
`model.architecture`. Tensor names (under `model.yolo` or `model.rfdetr`) and
FP32/FP16 precision are configurable. The input is always `[1, 3, H, W]` RGB
normalized to `[0, 1]`.

| `model.architecture` | Resize | Outputs |
| --- | --- | --- |
| `yolo` (YOLOv8/v11) | Letterbox | Detections `cx, cy, w, h, class scores..., mask coefficients...` and mask prototypes |
| `yolo26` | Letterbox | Detections `x1, y1, x2, y2, confidence, class_id, mask coefficients...` and mask prototypes |
| `rfdetr` (RF-DETR-Seg) | Stretch | Boxes `[1, queries, 4]` (normalized `cx, cy, w, h`), class logits `[1, queries, classes(+1)]`, mask logits `[1, queries, h, w]` |

YOLO detections are `[1, channels, anchors]` or `[1, anchors, channels]` and
prototypes `[1, mask_channels, mask_height, mask_width]`. RF-DETR yields one
detection per query and class above the confidence threshold. For models that
perform additional resizing and normalization inside the graph, set the node
input size to the model's exported input resolution.

Output dimensions must stay fixed for the configured input size. The node
resolves dynamic tensor metadata with an inference request during startup.
`model.class_names` follows model-output class order; `model.class_ids` contains
the corresponding semantic mask IDs in `1..65535`.

The `triton_server_url` launch argument sets the endpoint, defaults to
`triton-server:8001`, and takes precedence over the YAML value.
Shared memory (`prediction.use_shm`) requires a shared POSIX IPC namespace and
Triton client shared-memory access.

## Tracking and throughput

Tracking uses a separate ByteTrack tracker per camera, configured under
`tracking`. YOLOv8/v11 and RF-DETR detections use per-class NMS; YOLO26 uses
model selection.
Cameras run concurrently with one Triton client each. Busy cameras drop frames
at the admission limit. Set `prediction.max_in_flight_per_camera: 1` for low
latency; tracking enforces this limit to preserve frame order.

Enable `debug.print_timing` for processing times. To inspect results, enable
the `perception.publish_*` object lists and view their image observations with
the `perception_msgs/ImageObservations` RViz display from
[perception_interfaces](https://github.com/ika-rwth-aachen/perception_interfaces).
Use `ros_tracing:=true` to enable OpenADS dual-session tracing. Service images
include the `jazzy-ika` tracing fork.

Postprocessing thresholds and `debug.*` settings support runtime changes through
`ros2 param set`. Model, camera, and tracker settings are read-only after startup.
`ros2 param describe` exposes parameter descriptions and declared ranges.

## Diagnostics

`/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`) is remapped with
`diagnostics_topic`. Reports contain one `<FQN>: health` and optional
`<FQN>: <resolved_topic> topic status` entries. `diagnostic_updater.use_fqn: true`
includes the namespace (for example `/perception/image_instance_segmentation: health`)
without renaming topics. Hardware ID is `none`.

Health combines the same cycle's topic results and the checks below, with precedence
ERROR > STALE > WARN > OK. Check summaries appear alphabetically, followed by details.

| Check | Assessment | Detail fields per camera |
| --- | --- | --- |
| `inputs` | ERROR for receive timeout or acquisition age outside bounds; STALE before the first image until timeout | `received_since_start`, `receive_age_s`, `acquisition_age_s` |
| `data_integrity` | ERROR for zero dimensions, empty data/frame ID, or invalid/zero timestamp; STALE before input | `invalid_metadata_since_start` |
| `processing` | ERROR for exceptions, non-finite candidates or stalled output; WARN for slow processing; STALE pending backend/first result | `backend_ready`, `completed_since_start`, `failed_since_start`, `dropped_busy_since_start`, `dropped_not_ready_since_start`, `in_flight`, `output_age_s`, `duration_s`, `peak_duration_since_report_s` |
| `initialization` | Present after an initialization failure: ERROR until recovery, then OK | `error` |
| `topic.<camera>.<output>` | Enabled topic helper's frequency and timestamp assessment | Frequency, event-window and timestamp statistics |

Camera details use `<check>.<camera>.<field>`, such as
`processing.front_center.duration_s`. Errors and duration peaks remain visible
through the next assessment and recover with valid frames. Empty detections are
valid; busy/unavailable-backend drops are counted separately from successful output.

All settings below have the prefix `diagnostic_updater.` and are read-only.

| Parameter suffix | Default |
| --- | --- |
| `period` | `1.0` s |
| `use_fqn` | `true` (required) |
| `inputs.timeout` | `2.0` s |
| `inputs.min_age` / `inputs.max_age` | `-0.05` / `1.0` s |
| `processing.timeout` | `max(5.0, 2 * prediction.triton_client_timeout_s)` s |
| `processing.warn_duration` | `0.1` s, input callback to publication |
| `<camera>.<output>.nominal_frequency` | `0.0` Hz, disabling that topic check |
| `<camera>.<output>.min_frequency` / `max_frequency` | `0.8` / `1.2` times nominal frequency |

`<output>` is `input`, `class_mask`, `instance_mask` or `detections`; perception
object lists are demand-driven. Rate checks use five reports and the ROS clock. Output timestamp
limits add `processing.timeout` to `inputs.max_age`. Topic details under
`topic.<camera>.<output>` are `frequency_hz`, `{min,max,target}_frequency_hz`,
`events_in_window`, `events_since_start`, `window_s`, `{min,max}_delay_s`,
`{min,max}_allowed_delay_s` and `{late,early,zero_stamp}_reports_since_start`
(counts of reports observing timestamp faults).

Acquisition age uses ROS time (`use_sim_time` for bags); watchdogs use steady time.
Reports pause with `/clock` and assess elapsed time on resume. Backward time jumps
affect rate windows until they refill. Transport decoding failures appear as receive
timeouts. Backend readiness comes from initialization/inference responses, without
an independent probe. Device/time-sync state and mask-pixel validity are not assessed.
Application QoS is BEST_EFFORT/VOLATILE; RELIABLE subscribers do not match the outputs.

The `karl-platform` group `/perception/image_instance_segmentation` retains stopped
nodes with a five-second timeout. Grouping is monitoring-only; reactions are separate.
