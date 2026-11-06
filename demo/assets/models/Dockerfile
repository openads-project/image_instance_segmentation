# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

ARG MODEL_SOURCE_IMAGE=ghcr.io/openads-project/image_instance_segmentation/model:v1.0.0
FROM ${MODEL_SOURCE_IMAGE} AS model

FROM scratch

ARG IMAGE_VERSION

LABEL org.opencontainers.image.source="https://github.com/openads-project/image_instance_segmentation" \
      org.opencontainers.image.description="RF-DETR-Seg Small Triton model repository for the image instance segmentation demo" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=model /models/rfdetr-seg-small/ /models/rfdetr-seg-small/
COPY rfdetr-seg-small/MODEL_LICENSE rfdetr-seg-small/NOTICE /models/rfdetr-seg-small/
