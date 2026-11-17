// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "image_instance_segmentation/ImageInstanceSegmentation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <perception_msgs_utils/object_access.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>

namespace image_instance_segmentation {
namespace {

/** @brief Copy ROI pixels directly into a packed ROS image. */
sensor_msgs::msg::Image roiImage(const cv::Mat& pixels, const std_msgs::msg::Header& header, const std::string& encoding) {
  sensor_msgs::msg::Image image;
  image.header = header;
  image.width = static_cast<std::uint32_t>(pixels.cols);
  image.height = static_cast<std::uint32_t>(pixels.rows);
  image.encoding = encoding;
  image.step = static_cast<std::uint32_t>(pixels.cols * pixels.elemSize());
  image.data.resize(static_cast<std::size_t>(image.step) * image.height);
  for (int row = 0; row < pixels.rows; ++row)
    std::memcpy(image.data.data() + static_cast<std::size_t>(row) * image.step, pixels.ptr(row), image.step);
  return image;
}

/** @brief Clip the detection rectangle to the source image before integer conversion. */
cv::Rect sourceRoi(const cv::Rect2f& box, const cv::Size& size) {
  const int left = static_cast<int>(std::clamp(std::floor(static_cast<double>(box.x)), 0.0, static_cast<double>(size.width)));
  const int top = static_cast<int>(std::clamp(std::floor(static_cast<double>(box.y)), 0.0, static_cast<double>(size.height)));
  const int right = static_cast<int>(
      std::clamp(std::ceil(static_cast<double>(box.x) + box.width), static_cast<double>(left), static_cast<double>(size.width)));
  const int bottom = static_cast<int>(
      std::clamp(std::ceil(static_cast<double>(box.y) + box.height), static_cast<double>(top), static_cast<double>(size.height)));
  return {left, top, right - left, bottom - top};
}

}  // namespace

void ImageInstanceSegmentation::configurePerceptionClasses() {
  using Classification = perception_msgs::msg::ObjectClassification;
  using RoadUser = perception_msgs::msg::HEXAMOTION;
  const std::map<std::string, PerceptionClass> classes{
      {"PEDESTRIAN", {RoadUser::MODEL_ID, Classification::PEDESTRIAN}},
      {"BICYCLE", {RoadUser::MODEL_ID, Classification::BICYCLE}},
      {"MOTORCYCLE", {RoadUser::MODEL_ID, Classification::MOTORCYCLE}},
      {"MOTORBIKE", {RoadUser::MODEL_ID, Classification::MOTORCYCLE}},
      {"CAR", {RoadUser::MODEL_ID, Classification::CAR}},
      {"VAN", {RoadUser::MODEL_ID, Classification::CAR}},
      {"UTILITY", {RoadUser::MODEL_ID, Classification::UTILITY}},
      {"TRUCK", {RoadUser::MODEL_ID, Classification::UTILITY}},
      {"TRAILER", {RoadUser::MODEL_ID, Classification::UTILITY}},
      {"TRAIN", {RoadUser::MODEL_ID, Classification::UTILITY}},
      {"BUS", {RoadUser::MODEL_ID, Classification::BUS}},
      {"ANIMAL", {RoadUser::MODEL_ID, Classification::ANIMAL}},
      {"VRU", {RoadUser::MODEL_ID, Classification::VRU}},
      {"MICRO", {RoadUser::MODEL_ID, Classification::MICRO}},
      {"TRAFFIC_LIGHT", {perception_msgs::msg::TRAFFICLIGHT::MODEL_ID, 0}},
      {"TRAFFIC_SIGN", {perception_msgs::msg::TRAFFICSIGN::MODEL_ID, 0}},
  };
  perception_classes_.assign(class_names_.size(), {});
  if (!publish_road_users_ && !publish_traffic_lights_ && !publish_traffic_signs_) return;
  for (std::size_t i = 0; i < class_names_.size(); ++i) {
    const auto entry = classes.find(class_names_[i]);
    if (entry != classes.end()) {
      perception_classes_[i] = entry->second;
    } else {
      RCLCPP_WARN(get_logger(), "Class '%s' has no perception mapping; publishing it only in segmentation outputs",
                  class_names_[i].c_str());
    }
  }
}

void ImageInstanceSegmentation::publishPerceptionObjects(const CameraRuntime& camera,
                                                         const std_msgs::msg::Header& header,
                                                         const cv::Mat& bgr_image,
                                                         const std::vector<Detection>& detections,
                                                         const std::vector<InstanceMask>& masks) const {
  using namespace perception_msgs::object_access;
  // Image observations require a valid source camera header.
  if (header.frame_id.empty() || header.stamp.sec < 0 || header.stamp.nanosec >= 1000000000u) return;
  const auto publish = [&](const rclcpp::Publisher<perception_msgs::msg::ObjectList>::SharedPtr& publisher,
                           std::uint8_t model_id) {
    if (!publisher || (publisher->get_subscription_count() == 0 && publisher->get_intra_process_subscription_count() == 0))
      return;
    auto objects = std::make_unique<perception_msgs::msg::ObjectList>();
    objects->header = header;
    for (std::size_t i = 0; i < detections.size(); ++i) {
      const auto& detection = detections[i];
      const auto& classification = perception_classes_.at(static_cast<std::size_t>(detection.class_index));
      if (classification.model_id != model_id) continue;
      perception_msgs::msg::Object object;
      object.id = detection.object_id;
      if (!detection.is_tracked) {
        object.id = next_untracked_object_id_.fetch_add(1, std::memory_order_relaxed);
        if (object.id == 0) throw std::overflow_error("Untracked object ID exhausted");
      }
      object.existence_probability = detection.score;
      initializeState(object, model_id, CONTINUOUS_STATE_UNMEASURED);
      object.state.header = header;
      object.state.sensor_id.push_back(static_cast<std::uint64_t>(camera.config.sensor_id));
      if (model_id == perception_msgs::msg::HEXAMOTION::MODEL_ID) {
        perception_msgs::msg::ObjectClassification road_class;
        road_class.type = classification.classification;
        road_class.probability = detection.score;
        object.state.classifications.push_back(road_class);
      } else if (model_id == perception_msgs::msg::TRAFFICLIGHT::MODEL_ID) {
        setTrafficLightType(object, perception_msgs::msg::TRAFFICLIGHT::TYPE_UNKNOWN);
      }
      perception_msgs::msg::ImageObservation image;
      image.header = header;
      image.sensor_id = static_cast<std::uint64_t>(camera.config.sensor_id);
      image.image_width = static_cast<std::uint32_t>(bgr_image.cols);
      image.image_height = static_cast<std::uint32_t>(bgr_image.rows);
      image.bounding_box.center.position.x = detection.box.x + detection.box.width * 0.5;
      image.bounding_box.center.position.y = detection.box.y + detection.box.height * 0.5;
      image.bounding_box.size_x = detection.box.width;
      image.bounding_box.size_y = detection.box.height;
      const cv::Rect roi = i < masks.size() && !masks[i].roi.empty() ? masks[i].roi : sourceRoi(detection.box, bgr_image.size());
      if (!roi.empty()) {
        image.roi.x_offset = static_cast<std::uint32_t>(roi.x);
        image.roi.y_offset = static_cast<std::uint32_t>(roi.y);
        image.roi.width = static_cast<std::uint32_t>(roi.width);
        image.roi.height = static_cast<std::uint32_t>(roi.height);
        if (perception_include_crops_) image.crop = roiImage(bgr_image(roi), header, sensor_msgs::image_encodings::BGR8);
        if (perception_include_masks_ && i < masks.size() && !masks[i].pixels.empty())
          image.mask = roiImage(masks[i].pixels, header, sensor_msgs::image_encodings::MONO8);
      }
      object.state.image_observations.push_back(std::move(image));
      objects->objects.push_back(std::move(object));
    }
    publisher->publish(std::move(objects));
  };
  publish(camera.road_users_publisher, perception_msgs::msg::HEXAMOTION::MODEL_ID);
  publish(camera.traffic_lights_publisher, perception_msgs::msg::TRAFFICLIGHT::MODEL_ID);
  publish(camera.traffic_signs_publisher, perception_msgs::msg::TRAFFICSIGN::MODEL_ID);
}

}  // namespace image_instance_segmentation
