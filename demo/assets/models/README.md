# Model Image

Build the Triton model image mounted by the demo at `/models`. The Dockerfile
copies `rfdetr-seg-small` from the exported model image and adds
its license and attribution. Weights remain in container images, outside Git.

From the repository root:

```bash
MODEL_VERSION=v1.0.0
MODEL_IMAGE="ghcr.io/openads-project/image_instance_segmentation/model:${MODEL_VERSION}"

docker build \
  --build-arg IMAGE_VERSION="${MODEL_VERSION}" \
  --tag "${MODEL_IMAGE}" \
  demo/assets/models
```

`MODEL_SOURCE_IMAGE` selects the source model image and defaults to
`ghcr.io/openads-project/image_instance_segmentation/model:v1.0.0`. Its model
repository must contain `1/model.onnx`, `config.pbtxt`, and `model_manifest.yml`.

To test the result with the demo:

```bash
cd demo
MODEL_IMAGE="${MODEL_IMAGE}" docker compose up
```

To publish it, log in with your GitHub username and a token with `write:packages`,
then push:

```bash
docker login ghcr.io
docker push "${MODEL_IMAGE}"
```

The demo's `MODEL_IMAGE` setting selects the model image independently of
`SEGMENTATION_IMAGE`, which selects the ROS node image.

## Licensing

The model is distributed under Apache-2.0, matching the RF-DETR-Seg Small
code and pretrained segmentation weights. See Roboflow's
[model licensing](https://roboflow.com/model-licenses/rf-detr),
[MODEL_LICENSE](rfdetr-seg-small/MODEL_LICENSE), and
[NOTICE](rfdetr-seg-small/NOTICE).

The fine-tuned model retains upstream attribution and records ika's training
and export modifications. Preserve any additional notices supplied with the
source checkpoint. Rights to the training data and resulting model must permit
this distribution; the model license does not license the training dataset.
