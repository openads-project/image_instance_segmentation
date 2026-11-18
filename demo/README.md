# RF-DETR-Seg Demo

Four services run RF-DETR-Seg inference on a recorded camera stream: Triton,
rosbag playback, image instance segmentation, and RViz.

Requires Docker Compose 2.35 or newer, NVIDIA Container Toolkit with a compatible
GPU/driver, and an X11 or XWayland display.

Start the demo:

```bash
cd demo
xhost +local:
docker compose up -d
```

The bag comes from
`ghcr.io/openads-project/openadstack/demo-data:basic_v1.0.0` and is mounted
read-only. Its front-center camera's compressed images are replayed with a
playback clock, looping until stopped. Only the selected camera topic is replayed.
`DATA_IMAGE` selects the data image, `BAG_IMAGE_TOPIC` selects its
`sensor_msgs/CompressedImage` topic, and `BAG_RATE` sets the playback speed.
The data image must contain the bag under `/data`.

To customize the demo, copy `.env.example` to `.env` and edit the settings.

`SEGMENTATION_IMAGE` selects the segmentation node image; `MODEL_IMAGE` selects
the Triton model image. Model and node settings are in
[config/params.yml](config/params.yml).

RViz shows the image-only object lists `/demo/perception/road_users`,
`/demo/perception/traffic_signs`, and `/demo/perception/traffic_lights` in
separate panels: road users with boxes, masks, and track IDs on the matching
camera image, traffic signs and lights as their image crops. Panels can be
rearranged or floated as windows; the layout is in
[visualization/demo.rviz](visualization/demo.rviz). Full-frame masks and boxes
are published on `/demo/class_mask`, `/demo/instance_mask`, and
`/demo/detections`. No calibration or TF is needed for these image-coordinate
outputs.

The first start can take several minutes while TensorRT prepares the model.
Its cache is retained in a Docker volume for subsequent runs.

Stop the demo, remove its containers, and revoke display access:

```bash
docker compose down
xhost -local:
```

Use `docker compose down -v` to remove the TensorRT cache as well.

Model image build, licensing, and publishing instructions are in
[assets/models](assets/models/README.md).
