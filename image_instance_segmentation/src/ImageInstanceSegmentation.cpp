// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "image_instance_segmentation/ImageInstanceSegmentation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace image_instance_segmentation {

class ITracker {
 public:
  /** @brief Release the polymorphic tracker through its interface. */
  virtual ~ITracker() = default;
  /** @brief Associate the current detections and return tracks observed in this
   * frame. */
  virtual std::vector<Detection> update(const std::vector<Detection>& detections) = 0;
  /** @brief Discard track state and release reusable mask IDs. */
  virtual void reset() = 0;
};

ImageInstanceSegmentation::CameraRuntime::CameraRuntime() = default;
ImageInstanceSegmentation::CameraRuntime::~CameraRuntime() = default;

ImageInstanceSegmentation::CameraRuntime::CameraRuntime(CameraRuntime&& other) noexcept
    : config(std::move(other.config)),
      image_subscriber(std::move(other.image_subscriber)),
      class_mask_publisher(std::move(other.class_mask_publisher)),
      instance_mask_publisher(std::move(other.instance_mask_publisher)),
      detections_publisher(std::move(other.detections_publisher)),
      road_users_publisher(std::move(other.road_users_publisher)),
      traffic_lights_publisher(std::move(other.traffic_lights_publisher)),
      traffic_signs_publisher(std::move(other.traffic_signs_publisher)),
      callback_group(std::move(other.callback_group)),
      tracker(std::move(other.tracker)),
      in_flight(other.in_flight.load()) {}

ImageInstanceSegmentation::CameraRuntime& ImageInstanceSegmentation::CameraRuntime::operator=(CameraRuntime&& other) noexcept {
  if (this != &other) {
    config = std::move(other.config);
    image_subscriber = std::move(other.image_subscriber);
    class_mask_publisher = std::move(other.class_mask_publisher);
    instance_mask_publisher = std::move(other.instance_mask_publisher);
    detections_publisher = std::move(other.detections_publisher);
    road_users_publisher = std::move(other.road_users_publisher);
    traffic_lights_publisher = std::move(other.traffic_lights_publisher);
    traffic_signs_publisher = std::move(other.traffic_signs_publisher);
    callback_group = std::move(other.callback_group);
    tracker = std::move(other.tracker);
    in_flight.store(other.in_flight.load());
  }
  return *this;
}

// Fixed-size pool of persistent worker threads. Replaces spawning a new
// std::thread per frame, which under sustained camera frame rates repeatedly
// pays thread create/teardown cost for no benefit. Each camera has its own
// Triton client and mutex, so cameras run concurrently here; only same-camera
// in-flight requests (up to max_in_flight_per_camera_) serialize against each
// other.
class ThreadPool {
 public:
  /** @brief Start a fixed number of workers to consume queued inference tasks.
   */
  explicit ThreadPool(std::size_t worker_count) {
    worker_count = std::max<std::size_t>(1, worker_count);
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  /** @brief Prevent copying worker threads and their synchronization state. */
  ThreadPool(const ThreadPool&) = delete;
  /** @brief Prevent copy assignment of worker ownership. */
  ThreadPool& operator=(const ThreadPool&) = delete;

  /** @brief Drain queued work and join all workers before destroying shared
   * state. */
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  /** @brief Queue a task and wake one worker. */
  void enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push_back(std::move(task));
    }
    condition_.notify_one();
  }

 private:
  /** @brief Consume tasks until shutdown and the queue are both complete. */
  void workerLoop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (tasks_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool stopping_ = false;
};

namespace {

enum class TrackLifecycle {
  Tentative,
  Tracked,
  Lost,
  Removed,
};

/** @brief Map a supported tensor precision to its Triton dtype name. */
std::string tensorDataTypeName(TensorDataType data_type) {
  switch (data_type) {
    case TensorDataType::Fp32:
      return "fp32";
    case TensorDataType::Fp16:
      return "fp16";
  }
  return "unknown";
}

/** @brief Parse a configured precision name, rejecting unsupported dtypes. */
TensorDataType parseTensorDataType(const std::string& value) {
  if (value == "fp32" || value == "float32") {
    return TensorDataType::Fp32;
  }
  if (value == "fp16" || value == "float16" || value == "half") {
    return TensorDataType::Fp16;
  }
  throw std::runtime_error("unsupported tensor data type '" + value + "'");
}

/** @brief Parse a configured model architecture name. */
ModelArchitecture parseModelArchitecture(const std::string& value) {
  if (value == "yolo") return ModelArchitecture::Yolo;
  if (value == "yolo26") return ModelArchitecture::Yolo26;
  if (value == "rfdetr") return ModelArchitecture::RfDetr;
  throw std::runtime_error("unsupported model architecture '" + value + "'");
}

/** @brief Call @p function with a value of the scalar type matching @p data_type. */
template <typename Function>
void visitTensorDataType(TensorDataType data_type, Function&& function) {
  switch (data_type) {
    case TensorDataType::Fp32:
      function(float{});
      break;
    case TensorDataType::Fp16:
      function(Eigen::half{});
      break;
  }
}

/** @brief Map a tensor scalar type to the OpenCV element type. */
template <typename Scalar>
constexpr int cvTypeFor() {
  return std::is_same_v<Scalar, float> ? CV_32F : CV_16F;
}

/** @brief Return box intersection over union, or zero for a degenerate union.
 */
float intersectionOverUnion(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float intersection_area = (a & b).area();
  const float union_area = a.area() + b.area() - intersection_area;
  if (union_area <= 0.0f) {
    return 0.0f;
  }
  return intersection_area / union_area;
}

/** @brief Clamp a box to the image bounds and discard negative extents. */
cv::Rect2f clipBox(const cv::Rect2f& box, int width, int height) {
  const float x0 = std::max(0.0f, std::min(box.x, static_cast<float>(width - 1)));
  const float y0 = std::max(0.0f, std::min(box.y, static_cast<float>(height - 1)));
  const float x1 = std::max(0.0f, std::min(box.x + box.width, static_cast<float>(width)));
  const float y1 = std::max(0.0f, std::min(box.y + box.height, static_cast<float>(height)));
  return {x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)};
}

/** @brief Keep the box origin and ensure nonnegative width and height. */
cv::Rect2f sanitizeBox(const cv::Rect2f& box) {
  const float width = std::max(0.0f, box.width);
  const float height = std::max(0.0f, box.height);
  return {box.x, box.y, width, height};
}

/** @brief Return the integer pixel ROI covered by a box, clipped to the image. */
cv::Rect boxRoi(const cv::Rect2f& box, int width, int height) {
  return cv::Rect(static_cast<int>(std::floor(box.x)), static_cast<int>(std::floor(box.y)),
                  static_cast<int>(std::ceil(box.width)), static_cast<int>(std::ceil(box.height))) &
         cv::Rect(0, 0, width, height);
}

/** @brief Upsample low-resolution mask logits covering the image and threshold
 * them inside the detection box.
 *
 * Logits are interpolated before thresholding at logit(@p probability_threshold),
 * like the reference implementations; interpolating sigmoid probabilities instead
 * shifts the boundaries of upsampled masks. */
InstanceMask maskFromLogits(
    const cv::Mat& logits, const cv::Rect2f& box, int image_width, int image_height, double probability_threshold) {
  InstanceMask mask{boxRoi(box, image_width, image_height), {}};
  if (mask.roi.empty()) return mask;
  cv::Mat resized;
  cv::resize(logits, resized, cv::Size(image_width, image_height), 0.0, 0.0, cv::INTER_LINEAR);
  const double logit_threshold = std::log(probability_threshold / (1.0 - probability_threshold));
  cv::compare(resized(mask.roi), logit_threshold, mask.pixels, cv::CMP_GT);
  return mask;
}

/** @brief Validate image storage and convert supported encodings to BGR; BGR
 * input shares storage. */
cv::Mat rawImageToBgr(const sensor_msgs::msg::Image& msg) {
  namespace enc = sensor_msgs::image_encodings;

  if (msg.width == 0 || msg.height == 0 || msg.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      msg.height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("image dimensions must be positive and fit into OpenCV dimensions");
  }

  auto checkStep = [&](std::size_t minimum_step) {
    if (msg.step < minimum_step) {
      throw std::runtime_error("image step is smaller than width * pixel size");
    }
    if (msg.data.size() < static_cast<std::size_t>(msg.step) * msg.height) {
      throw std::runtime_error("image data buffer is smaller than step * height");
    }
  };

  if (msg.encoding == enc::BGR8 || msg.encoding == "8UC3") {
    checkStep(static_cast<std::size_t>(msg.width) * 3);
    return cv::Mat(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3, const_cast<std::uint8_t*>(msg.data.data()),
                   msg.step);
  }
  if (msg.encoding == enc::RGB8) {
    checkStep(static_cast<std::size_t>(msg.width) * 3);
    cv::Mat rgb(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3, const_cast<std::uint8_t*>(msg.data.data()),
                msg.step);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  if (msg.encoding == enc::BGRA8 || msg.encoding == "8UC4") {
    checkStep(static_cast<std::size_t>(msg.width) * 4);
    cv::Mat bgra(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4, const_cast<std::uint8_t*>(msg.data.data()),
                 msg.step);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
  }
  if (msg.encoding == enc::RGBA8) {
    checkStep(static_cast<std::size_t>(msg.width) * 4);
    cv::Mat rgba(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4, const_cast<std::uint8_t*>(msg.data.data()),
                 msg.step);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
  }
  if (msg.encoding == enc::MONO8 || msg.encoding == "8UC1") {
    checkStep(static_cast<std::size_t>(msg.width));
    cv::Mat mono(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC1, const_cast<std::uint8_t*>(msg.data.data()),
                 msg.step);
    cv::Mat bgr;
    cv::cvtColor(mono, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }
  if (msg.encoding == enc::MONO16 || msg.encoding == "16UC1") {
    checkStep(static_cast<std::size_t>(msg.width) * 2);
    cv::Mat mono16(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_16UC1,
                   const_cast<std::uint8_t*>(msg.data.data()), msg.step);
    const uint16_t endian_marker = 1;
    const bool host_big_endian = *reinterpret_cast<const uint8_t*>(&endian_marker) == 0;
    if (static_cast<bool>(msg.is_bigendian) != host_big_endian) {
      cv::Mat native(mono16.size(), CV_16UC1);
      for (int y = 0; y < native.rows; ++y) {
        const auto* source = msg.data.data() + static_cast<std::size_t>(y) * msg.step;
        auto* destination = native.ptr<uint16_t>(y);
        for (int x = 0; x < native.cols; ++x) {
          destination[x] = msg.is_bigendian ? (static_cast<uint16_t>(source[2 * x]) << 8) | source[2 * x + 1]
                                            : (static_cast<uint16_t>(source[2 * x + 1]) << 8) | source[2 * x];
        }
      }
      mono16 = std::move(native);
    }
    cv::Mat mono8;
    mono16.convertTo(mono8, CV_8UC1, 1.0 / 256.0);
    cv::Mat bgr;
    cv::cvtColor(mono8, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }

  throw std::runtime_error("unsupported raw image encoding '" + msg.encoding + "'");
}

/** @brief Encode a box as center x, center y, aspect ratio, and height for
 * Kalman correction. */
cv::Vec<float, 4> measurementFromBox(const cv::Rect2f& box) {
  const float safe_height = std::max(box.height, 1.0f);
  const float center_x = box.x + box.width * 0.5f;
  const float center_y = box.y + box.height * 0.5f;
  const float aspect_ratio = box.width / safe_height;
  return {center_x, center_y, aspect_ratio, safe_height};
}

/** @brief Convert the Kalman position state into a box with positive aspect
 * ratio and height. */
cv::Rect2f boxFromState(const cv::Mat& state) {
  const float center_x = state.at<float>(0, 0);
  const float center_y = state.at<float>(1, 0);
  const float aspect_ratio = std::max(state.at<float>(2, 0), 1e-3f);
  const float height = std::max(state.at<float>(3, 0), 1.0f);
  const float width = std::max(aspect_ratio * height, 1.0f);
  return sanitizeBox({center_x - width * 0.5f, center_y - height * 0.5f, width, height});
}

/** @brief Initialize the eight-state constant-velocity box filter from an
 * observation. */
cv::KalmanFilter makeKalmanFilter(const cv::Rect2f& box) {
  cv::KalmanFilter filter(8, 4, 0, CV_32F);
  filter.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
  for (int i = 0; i < 4; ++i) {
    filter.transitionMatrix.at<float>(i, i + 4) = 1.0f;
  }
  filter.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
  for (int i = 0; i < 4; ++i) {
    filter.measurementMatrix.at<float>(i, i) = 1.0f;
  }
  cv::setIdentity(filter.processNoiseCov, cv::Scalar(1e-2));
  cv::setIdentity(filter.measurementNoiseCov, cv::Scalar(1e-1));
  cv::setIdentity(filter.errorCovPost, cv::Scalar(1.0));
  filter.statePost = cv::Mat::zeros(8, 1, CV_32F);
  const auto measurement = measurementFromBox(box);
  for (int i = 0; i < 4; ++i) {
    filter.statePost.at<float>(i, 0) = measurement[i];
  }
  filter.statePre = filter.statePost.clone();
  return filter;
}

/** @brief Solve a rectangular minimum-cost assignment using a padded Hungarian
 * algorithm. */
std::vector<int> solveAssignment(const std::vector<std::vector<float>>& cost_matrix) {
  const int rows = static_cast<int>(cost_matrix.size());
  const int cols = rows > 0 ? static_cast<int>(cost_matrix.front().size()) : 0;
  if (rows == 0 || cols == 0) {
    return std::vector<int>(rows, -1);
  }

  const int size = std::max(rows, cols);
  const double kPadCost = 1e3;
  std::vector<std::vector<double>> cost(size + 1, std::vector<double>(size + 1, kPadCost));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      cost[i + 1][j + 1] = static_cast<double>(cost_matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
    }
  }

  std::vector<double> u(size + 1, 0.0);
  std::vector<double> v(size + 1, 0.0);
  std::vector<int> p(size + 1, 0);
  std::vector<int> way(size + 1, 0);

  for (int i = 1; i <= size; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(size + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(size + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = std::numeric_limits<double>::infinity();
      int j1 = 0;
      for (int j = 1; j <= size; ++j) {
        if (used[j]) {
          continue;
        }
        const double cur = cost[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= size; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(rows, -1);
  for (int j = 1; j <= size; ++j) {
    if (p[j] >= 1 && p[j] <= rows && j <= cols) {
      assignment[static_cast<std::size_t>(p[j] - 1)] = j - 1;
    }
  }
  return assignment;
}

class ByteTracker final : public ITracker {
 public:
  /** @brief Initialize a per-camera tracker with the configured association
   * thresholds. */
  explicit ByteTracker(const TrackerConfig& config) : config_(config) {}

  /** @brief Predict tracks, associate high then low confidence observations,
   * and retire expired tracks. */
  std::vector<Detection> update(const std::vector<Detection>& detections) override {
    for (auto& track : tracks_) {
      if (track.state == TrackLifecycle::Removed) {
        continue;
      }
      track.predicted_box = boxFromState(track.filter.predict());
      track.updated_this_frame = false;
    }

    std::vector<int> high_detection_indices;
    std::vector<int> low_detection_indices;
    high_detection_indices.reserve(detections.size());
    low_detection_indices.reserve(detections.size());
    for (std::size_t i = 0; i < detections.size(); ++i) {
      const auto& detection = detections[i];
      if (detection.box.area() < static_cast<float>(config_.min_box_area_px)) {
        continue;
      }
      if (detection.score >= static_cast<float>(config_.high_confidence_threshold)) {
        high_detection_indices.push_back(static_cast<int>(i));
      } else if (detection.score >= static_cast<float>(config_.low_confidence_threshold)) {
        low_detection_indices.push_back(static_cast<int>(i));
      }
    }

    std::vector<int> candidate_track_indices;
    candidate_track_indices.reserve(tracks_.size());
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
      if (tracks_[i].state != TrackLifecycle::Removed) {
        candidate_track_indices.push_back(static_cast<int>(i));
      }
    }

    auto first_stage = associate(candidate_track_indices, high_detection_indices, detections);
    for (const auto& match : first_stage.matches) {
      updateTrack(tracks_[static_cast<std::size_t>(match.first)], detections[static_cast<std::size_t>(match.second)]);
    }

    std::vector<int> second_stage_tracks;
    second_stage_tracks.reserve(first_stage.unmatched_track_indices.size());
    for (int track_index : first_stage.unmatched_track_indices) {
      const auto state = tracks_[static_cast<std::size_t>(track_index)].state;
      if (state == TrackLifecycle::Tracked || state == TrackLifecycle::Lost) {
        second_stage_tracks.push_back(track_index);
      }
    }

    auto second_stage = associate(second_stage_tracks, low_detection_indices, detections);
    for (const auto& match : second_stage.matches) {
      updateTrack(tracks_[static_cast<std::size_t>(match.first)], detections[static_cast<std::size_t>(match.second)]);
    }

    for (int detection_index : first_stage.unmatched_detection_indices) {
      const auto& detection = detections[static_cast<std::size_t>(detection_index)];
      if (detection.score >= static_cast<float>(config_.new_track_confidence_threshold)) {
        createTrack(detection);
      }
    }

    std::vector<Detection> tracked_detections;
    for (auto& track : tracks_) {
      if (track.state == TrackLifecycle::Removed) {
        continue;
      }
      if (track.updated_this_frame) {
        if (track.state == TrackLifecycle::Tentative && track.hits >= 2) {
          track.state = TrackLifecycle::Tracked;
        }
        track.latest_detection.track_id = track.id;
        track.latest_detection.object_id = track.object_id;
        track.latest_detection.is_tracked = true;
        tracked_detections.push_back(track.latest_detection);
        continue;
      }

      if (track.state == TrackLifecycle::Tentative) {
        retireTrack(track);
        continue;
      }

      ++track.lost_frames;
      if (track.lost_frames > config_.max_lost_frames) {
        retireTrack(track);
      } else {
        track.state = TrackLifecycle::Lost;
      }
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), [](const Track& track) { return track.state == TrackLifecycle::Removed; }),
        tracks_.end());

    std::sort(tracked_detections.begin(), tracked_detections.end(),
              [](const Detection& lhs, const Detection& rhs) { return lhs.track_id < rhs.track_id; });
    return tracked_detections;
  }

  /** @brief Clear all tracks and reset the recyclable mono16 mask-ID pool. */
  void reset() override {
    tracks_.clear();
    reusable_ids_.clear();
    next_track_id_ = 1;
  }

 private:
  struct Track {
    cv::KalmanFilter filter;
    cv::Rect2f predicted_box;
    Detection latest_detection;
    TrackLifecycle state = TrackLifecycle::Tentative;
    std::uint16_t id = 0;
    std::uint64_t object_id = 0;
    int hits = 0;
    int lost_frames = 0;
    bool updated_this_frame = false;
  };

  struct AssociationResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_track_indices;
    std::vector<int> unmatched_detection_indices;
  };

  /** @brief Match class-compatible boxes above the IoU threshold and return
   * unmatched indices. */
  AssociationResult associate(const std::vector<int>& track_indices,
                              const std::vector<int>& detection_indices,
                              const std::vector<Detection>& detections) const {
    AssociationResult result;
    if (track_indices.empty()) {
      result.unmatched_detection_indices = detection_indices;
      return result;
    }
    if (detection_indices.empty()) {
      result.unmatched_track_indices = track_indices;
      return result;
    }

    const float max_cost = 1.0f - static_cast<float>(config_.match_iou_threshold);
    const float blocked_cost = 1e3f;
    std::vector<std::vector<float>> cost_matrix(track_indices.size(), std::vector<float>(detection_indices.size(), blocked_cost));

    for (std::size_t row = 0; row < track_indices.size(); ++row) {
      const auto& track = tracks_[static_cast<std::size_t>(track_indices[row])];
      for (std::size_t col = 0; col < detection_indices.size(); ++col) {
        const auto& detection = detections[static_cast<std::size_t>(detection_indices[col])];
        if (track.latest_detection.class_index != detection.class_index) {
          continue;
        }
        const float iou = intersectionOverUnion(track.predicted_box, detection.box);
        if (iou < static_cast<float>(config_.match_iou_threshold)) {
          continue;
        }
        cost_matrix[row][col] = 1.0f - iou;
      }
    }

    const auto assignment = solveAssignment(cost_matrix);
    std::vector<bool> detection_taken(detection_indices.size(), false);
    for (std::size_t row = 0; row < track_indices.size(); ++row) {
      const int assigned_col = assignment[row];
      if (assigned_col >= 0 && assigned_col < static_cast<int>(detection_indices.size()) &&
          cost_matrix[row][static_cast<std::size_t>(assigned_col)] <= max_cost) {
        result.matches.emplace_back(track_indices[row], detection_indices[static_cast<std::size_t>(assigned_col)]);
        detection_taken[static_cast<std::size_t>(assigned_col)] = true;
      } else {
        result.unmatched_track_indices.push_back(track_indices[row]);
      }
    }

    for (std::size_t col = 0; col < detection_indices.size(); ++col) {
      if (!detection_taken[col]) {
        result.unmatched_detection_indices.push_back(detection_indices[col]);
      }
    }
    return result;
  }

  /** @brief Correct the track filter and refresh its latest detection and
   * lifecycle counters. */
  void updateTrack(Track& track, const Detection& detection) {
    cv::Mat measurement(4, 1, CV_32F);
    const auto measurement_values = measurementFromBox(detection.box);
    for (int i = 0; i < 4; ++i) {
      measurement.at<float>(i, 0) = measurement_values[i];
    }
    track.filter.correct(measurement);
    track.predicted_box = boxFromState(track.filter.statePost);
    track.latest_detection = detection;
    track.latest_detection.box = track.predicted_box;
    track.latest_detection.track_id = track.id;
    track.latest_detection.object_id = track.object_id;
    track.latest_detection.is_tracked = true;
    track.updated_this_frame = true;
    track.lost_frames = 0;
    ++track.hits;
    if (track.state == TrackLifecycle::Lost || track.state == TrackLifecycle::Tracked) {
      track.state = TrackLifecycle::Tracked;
    }
  }

  /** @brief Allocate a mask ID and start a tentative Kalman track from a
   * detection. */
  void createTrack(const Detection& detection) {
    const std::uint16_t id = allocateTrackId();
    if (id == 0) {
      return;
    }

    Track track;
    track.filter = makeKalmanFilter(detection.box);
    track.predicted_box = detection.box;
    track.latest_detection = detection;
    if (next_object_id_ == 0) throw std::overflow_error("Object tracking ID exhausted");
    track.object_id = next_object_id_++;
    track.latest_detection.track_id = id;
    track.latest_detection.object_id = track.object_id;
    track.latest_detection.is_tracked = true;
    track.state = TrackLifecycle::Tentative;
    track.id = id;
    track.hits = 1;
    track.updated_this_frame = true;
    tracks_.push_back(std::move(track));
  }

  /** @brief Mark a track removed and return its mask ID to the reusable pool.
   */
  void retireTrack(Track& track) {
    if (track.id != 0) {
      reusable_ids_.push_back(track.id);
      track.id = 0;
    }
    track.state = TrackLifecycle::Removed;
    track.updated_this_frame = false;
  }

  /** @brief Reuse an available mono16 ID or allocate the next ID; return zero
   * on exhaustion. */
  std::uint16_t allocateTrackId() {
    if (!reusable_ids_.empty()) {
      const std::uint16_t id = reusable_ids_.front();
      reusable_ids_.pop_front();
      return id;
    }
    if (next_track_id_ == 0) {
      return 0;
    }
    return next_track_id_++;
  }

  TrackerConfig config_;
  std::vector<Track> tracks_;
  std::deque<std::uint16_t> reusable_ids_;
  std::uint16_t next_track_id_ = 1;
  std::uint64_t next_object_id_ = 1;
};

}  // namespace

ImageInstanceSegmentation::ImageInstanceSegmentation(const rclcpp::NodeOptions& options)
    : rclcpp::Node("image_instance_segmentation", options) {
  // Concurrency already comes from our own per-camera worker threads; letting
  // OpenCV additionally fan each resize/cvtColor/addWeighted call out across
  // all cores on top of that oversubscribes the CPU for no throughput gain.
  cv::setNumThreads(1);

  loadParameters();
  validateParameters();
  configurePerceptionClasses();
  initializeModel();
  setupCameras();
  setupDiagnostics();
  parameters_callback_handle_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter>& parameters) { return parametersCallback(parameters); });
  parameters_post_callback_handle_ =
      add_post_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters) { applyParameters(parameters); });
  const std::size_t worker_count = static_cast<std::size_t>(std::max(1, max_in_flight_per_camera_)) * camera_configs_.size();
  thread_pool_ = std::make_unique<ThreadPool>(worker_count);
  for (const auto& config : camera_configs_) {
    thread_pool_->enqueue([this, name = config.name] {
      try {
        initializeModelForCamera(name);
      } catch (const std::exception& error) {
        if (!stopping_.load()) recordProcessingFailure(name, error.what());
      }
    });
  }
}

ImageInstanceSegmentation::~ImageInstanceSegmentation() {
  // Unloading a component need not shut down the ROS context. Stop reconnect
  // retries explicitly before joining workers, while their camera state is
  // alive.
  {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    stopping_.store(true);
  }
  for (auto& entry : cameras_) {
    entry.second.image_subscriber.shutdown();
  }
  diagnostic_updater_.reset();
  thread_pool_.reset();
}

template <typename T>
void ImageInstanceSegmentation::declareAndLoadParameter(const std::string& name,
                                                        T& value,
                                                        const std::string& description,
                                                        bool add_to_auto_reconfigurable_params,
                                                        bool is_required,
                                                        bool read_only,
                                                        const std::optional<double>& from_value,
                                                        const std::optional<double>& to_value,
                                                        const std::optional<double>& step_value,
                                                        const std::string& additional_constraints) {
  if (add_to_auto_reconfigurable_params && read_only) {
    throw std::logic_error("A read-only parameter cannot be reconfigurable: " + name);
  }
  if (is_required && get_node_parameters_interface()->get_parameter_overrides().count(name) == 0) {
    throw std::runtime_error("Missing required parameter: " + name);
  }
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  descriptor.additional_constraints = additional_constraints;
  descriptor.read_only = read_only;
  if (from_value && to_value) {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      rcl_interfaces::msg::IntegerRange range;
      range.from_value = static_cast<int64_t>(*from_value);
      range.to_value = static_cast<int64_t>(*to_value);
      range.step = static_cast<uint64_t>(step_value.value_or(1.0));
      descriptor.integer_range = {range};
    } else if constexpr (std::is_floating_point_v<T>) {
      rcl_interfaces::msg::FloatingPointRange range;
      range.from_value = *from_value;
      range.to_value = *to_value;
      range.step = step_value.value_or(0.0);
      descriptor.floating_point_range = {range};
    }
  }
  value = declare_parameter<T>(name, value, descriptor);
  if (add_to_auto_reconfigurable_params) {
    auto_reconfigurable_params_.emplace(name, [&value](const rclcpp::Parameter& parameter) { value = parameter.get_value<T>(); });
  }
}

// Parameter types used by the diagnostics translation unit.
template void ImageInstanceSegmentation::declareAndLoadParameter<bool>(const std::string&,
                                                                       bool&,
                                                                       const std::string&,
                                                                       bool,
                                                                       bool,
                                                                       bool,
                                                                       const std::optional<double>&,
                                                                       const std::optional<double>&,
                                                                       const std::optional<double>&,
                                                                       const std::string&);
template void ImageInstanceSegmentation::declareAndLoadParameter<double>(const std::string&,
                                                                         double&,
                                                                         const std::string&,
                                                                         bool,
                                                                         bool,
                                                                         bool,
                                                                         const std::optional<double>&,
                                                                         const std::optional<double>&,
                                                                         const std::optional<double>&,
                                                                         const std::string&);

rcl_interfaces::msg::SetParametersResult ImageInstanceSegmentation::parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters) const {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto& parameter : parameters) {
    if (auto_reconfigurable_params_.count(parameter.get_name()) == 0) continue;
    if (parameter.get_type() != get_parameter_types({parameter.get_name()}).front()) {
      result.successful = false;
      result.reason = "Parameter type cannot be changed: " + parameter.get_name();
      break;
    }
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE && !std::isfinite(parameter.as_double())) {
      result.successful = false;
      result.reason = "Parameter must be finite: " + parameter.get_name();
      break;
    }
  }
  return result;
}

void ImageInstanceSegmentation::applyParameters(const std::vector<rclcpp::Parameter>& parameters) {
  std::unique_lock<std::shared_mutex> lock(runtime_parameters_mutex_);
  for (const auto& parameter : parameters) {
    const auto setter = auto_reconfigurable_params_.find(parameter.get_name());
    if (setter != auto_reconfigurable_params_.end()) setter->second(parameter);
  }
}

void ImageInstanceSegmentation::loadParameters() {
  declareAndLoadParameter("prediction.triton_server_url", server_url_, "Triton gRPC host and port");
  declareAndLoadParameter("prediction.model_name", model_name_, "Triton model name");
  declareAndLoadParameter("prediction.model_version", model_version_, "Triton model version");
  declareAndLoadParameter("prediction.triton_client_timeout_s", triton_client_timeout_s_, "Triton request timeout in seconds",
                          false, false, true, std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 0.0);
  declareAndLoadParameter("prediction.use_shm", use_shm_, "Use Triton shared memory transport");
  declareAndLoadParameter("prediction.max_in_flight_per_camera", max_in_flight_per_camera_,
                          "Maximum admitted frames per camera; tracking forces one", false, false, true, 1,
                          std::numeric_limits<int>::max(), 1);

  declareAndLoadParameter("model.architecture", architecture_name_, "Output decoding: yolo (YOLOv8/v11), yolo26, or rfdetr");
  architecture_ = parseModelArchitecture(architecture_name_);
  declareAndLoadParameter("model.input_width", input_width_, "Model input width in pixels", false, false, true, 1,
                          std::numeric_limits<int>::max(), 1);
  declareAndLoadParameter("model.input_height", input_height_, "Model input height in pixels", false, false, true, 1,
                          std::numeric_limits<int>::max(), 1);
  declareAndLoadParameter("model.input_tensor_name", input_tensor_name_, "Input tensor name");
  if (architecture_ == ModelArchitecture::RfDetr) {
    declareAndLoadParameter("model.rfdetr.boxes_output_tensor_name", rfdetr_boxes_output_tensor_name_, "RF-DETR box tensor name");
    declareAndLoadParameter("model.rfdetr.logits_output_tensor_name", rfdetr_logits_output_tensor_name_,
                            "RF-DETR class logit tensor name");
    declareAndLoadParameter("model.rfdetr.masks_output_tensor_name", rfdetr_masks_output_tensor_name_,
                            "RF-DETR mask tensor name");
  } else {
    declareAndLoadParameter("model.yolo.detections_output_tensor_name", yolo_detections_output_tensor_name_,
                            "YOLO detection tensor name");
    declareAndLoadParameter("model.yolo.prototypes_output_tensor_name", yolo_prototypes_output_tensor_name_,
                            "YOLO mask prototype tensor name");
  }
  declareAndLoadParameter("model.input_tensor_data_type", input_dtype_, "Triton input tensor dtype: fp32 or fp16");
  input_tensor_data_type_ = parseTensorDataType(input_dtype_);
  declareAndLoadParameter("model.output_tensor_data_type", output_dtype_, "Triton output tensor dtype: fp32 or fp16");
  output_tensor_data_type_ = parseTensorDataType(output_dtype_);
  declareAndLoadParameter("model.class_names", class_names_, "Class names in contiguous model-output order");
  declareAndLoadParameter("model.class_ids", class_ids_,
                          "Semantic IDs written into `class_mask`; zero is "
                          "reserved for background");

  declareAndLoadParameter("postprocessing.confidence_threshold", confidence_threshold_,
                          "Minimum detection confidence; tracking also uses its low-confidence threshold", true, false, false,
                          0.0, 1.0, 0.0);
  declareAndLoadParameter("postprocessing.mask_threshold", mask_threshold_, "Binary mask threshold", true, false, false, 0.0, 1.0,
                          0.0);
  declareAndLoadParameter("postprocessing.nms_iou_threshold", nms_iou_threshold_,
                          "Per-class NMS IoU threshold; YOLO26 selects detections in the model", true, false, false, 0.0, 1.0,
                          0.0);

  declareAndLoadParameter("tracking.enabled", tracker_config_.enabled, "Enable per-camera ByteTrack-based tracking");
  declareAndLoadParameter("tracking.backend", tracker_config_.backend, "Tracking backend; only `bytetrack` is supported");
  declareAndLoadParameter("tracking.high_confidence_threshold", tracker_config_.high_confidence_threshold,
                          "High-confidence association threshold", false, false, true, 0.0, 1.0, 0.0);
  declareAndLoadParameter("tracking.low_confidence_threshold", tracker_config_.low_confidence_threshold,
                          "Low-confidence association threshold", false, false, true, 0.0, 1.0, 0.0);
  declareAndLoadParameter("tracking.new_track_confidence_threshold", tracker_config_.new_track_confidence_threshold,
                          "Minimum score required to start a new track", false, false, true, 0.0, 1.0, 0.0);
  declareAndLoadParameter("tracking.match_iou_threshold", tracker_config_.match_iou_threshold,
                          "Minimum IoU used for tracker association", false, false, true, std::numeric_limits<double>::min(), 1.0,
                          0.0);
  declareAndLoadParameter("tracking.max_lost_frames", tracker_config_.max_lost_frames,
                          "Frames a track may remain unmatched before retirement", false, false, true, 1,
                          std::numeric_limits<int>::max(), 1);
  declareAndLoadParameter("tracking.min_box_area_px", tracker_config_.min_box_area_px,
                          "Minimum detection area eligible for tracking", false, false, true, 0.0,
                          std::numeric_limits<double>::max(), 0.0);

  declareAndLoadParameter("perception.publish_road_users", publish_road_users_, "Publish image-only road-user object lists");
  declareAndLoadParameter("perception.publish_traffic_lights", publish_traffic_lights_,
                          "Publish image-only traffic-light object lists");
  declareAndLoadParameter("perception.publish_traffic_signs", publish_traffic_signs_,
                          "Publish image-only traffic-sign object lists");
  declareAndLoadParameter("perception.include_crops", perception_include_crops_, "Include BGR image crops in perception objects");
  declareAndLoadParameter("perception.include_masks", perception_include_masks_,
                          "Include binary instance masks in perception objects");

  declareAndLoadParameter("debug.print_timing", print_timing_, "Log inference, postprocessing, and total processing times", true,
                          false, false);

  if (tracker_config_.enabled && max_in_flight_per_camera_ != 1) {
    RCLCPP_INFO(get_logger(),
                "Tracking is enabled; forcing prediction.max_in_flight_per_camera from "
                "%d to 1 for ordered per-camera processing.",
                max_in_flight_per_camera_);
    max_in_flight_per_camera_ = 1;
  }

  camera_configs_.clear();
  declareAndLoadParameter("cameras", camera_names_, "Names of cameras to process");
  for (const auto& name : camera_names_) {
    const std::string prefix = "camera." + name + ".";
    CameraConfig config;
    config.name = name;
    config.sensor_id = static_cast<int64_t>(camera_configs_.size()) + 1;
    if (name != "camera") {
      config.image_topic = "~/" + name + "/image";
      config.output_prefix = "~/" + name;
    }
    declareAndLoadParameter(prefix + "sensor_id", config.sensor_id, "Non-negative camera identifier in perception observations");
    declareAndLoadParameter(prefix + "image_topic", config.image_topic, "Raw image base topic");
    declareAndLoadParameter(prefix + "transport", config.transport, "Image transport hint: raw or compressed");
    declareAndLoadParameter(prefix + "output_prefix", config.output_prefix, "Prefix for this camera's output topics");
    camera_configs_.push_back(config);
  }
}

void ImageInstanceSegmentation::validateParameters() const {
  auto fail = [](const std::string& message) {
    throw std::runtime_error("Invalid image_instance_segmentation parameter: " + message);
  };
  if (max_in_flight_per_camera_ <= 0) fail("prediction.max_in_flight_per_camera must be positive");
  if (!std::isfinite(triton_client_timeout_s_) || triton_client_timeout_s_ <= 0.0) {
    fail("prediction.triton_client_timeout_s must be finite and positive");
  }
  if (server_url_.empty()) fail("prediction.triton_server_url must not be empty");
  if (model_name_.empty()) fail("prediction.model_name must not be empty");
  if (input_width_ <= 0 || input_height_ <= 0) fail("model input dimensions must be positive");
  for (const auto& name : outputTensorNames()) {
    if (name.empty()) fail("model output tensor names must not be empty");
  }
  if (input_tensor_data_type_ != TensorDataType::Fp32 && input_tensor_data_type_ != TensorDataType::Fp16) {
    fail("model.input_tensor_data_type must be either 'fp32' or 'fp16'");
  }
  if (output_tensor_data_type_ != TensorDataType::Fp32 && output_tensor_data_type_ != TensorDataType::Fp16) {
    fail("model.output_tensor_data_type must be either 'fp32' or 'fp16'");
  }
  if (class_names_.empty()) fail("model.class_names must contain at least one class");
  if (class_ids_.size() != class_names_.size()) fail("model.class_ids must have the same length as model.class_names");
  if (std::any_of(class_ids_.begin(), class_ids_.end(), [](int64_t id) { return id <= 0 || id > 65535; })) {
    fail(
        "model.class_ids entries must fit into mono16 masks and reserve zero "
        "for background");
  }
  if (!std::isfinite(confidence_threshold_) || confidence_threshold_ < 0.0 || confidence_threshold_ > 1.0)
    fail("confidence threshold must be in [0, 1]");
  if (!std::isfinite(mask_threshold_) || mask_threshold_ < 0.0 || mask_threshold_ > 1.0) fail("mask threshold must be in [0, 1]");
  if (!std::isfinite(nms_iou_threshold_) || nms_iou_threshold_ < 0.0 || nms_iou_threshold_ > 1.0)
    fail("NMS IoU threshold must be in [0, 1]");
  if (tracker_config_.backend != "bytetrack") fail("tracking.backend must be 'bytetrack'");
  if (!std::isfinite(tracker_config_.low_confidence_threshold) || tracker_config_.low_confidence_threshold < 0.0 ||
      tracker_config_.low_confidence_threshold > 1.0) {
    fail("tracking.low_confidence_threshold must be in [0, 1]");
  }
  if (!std::isfinite(tracker_config_.high_confidence_threshold) ||
      tracker_config_.high_confidence_threshold < tracker_config_.low_confidence_threshold ||
      tracker_config_.high_confidence_threshold > 1.0) {
    fail(
        "tracking.high_confidence_threshold must be in "
        "[tracking.low_confidence_threshold, 1]");
  }
  if (!std::isfinite(tracker_config_.new_track_confidence_threshold) ||
      tracker_config_.new_track_confidence_threshold < tracker_config_.high_confidence_threshold ||
      tracker_config_.new_track_confidence_threshold > 1.0) {
    fail(
        "tracking.new_track_confidence_threshold must be in "
        "[tracking.high_confidence_threshold, 1]");
  }
  if (!std::isfinite(tracker_config_.match_iou_threshold) || tracker_config_.match_iou_threshold <= 0.0 ||
      tracker_config_.match_iou_threshold > 1.0) {
    fail("tracking.match_iou_threshold must be in (0, 1]");
  }
  if (tracker_config_.max_lost_frames < 1) fail("tracking.max_lost_frames must be at least 1");
  if (!std::isfinite(tracker_config_.min_box_area_px) || tracker_config_.min_box_area_px < 0.0)
    fail("tracking.min_box_area_px must be non-negative");
  if (camera_configs_.empty()) fail("cameras must contain at least one camera name");
  std::vector<std::string> camera_names;
  std::vector<int64_t> sensor_ids;
  for (const auto& camera : camera_configs_) {
    if (std::find(camera_names.begin(), camera_names.end(), camera.name) != camera_names.end()) {
      fail("camera names must be unique");
    }
    camera_names.push_back(camera.name);
    if (camera.sensor_id < 0 || std::find(sensor_ids.begin(), sensor_ids.end(), camera.sensor_id) != sensor_ids.end())
      fail("camera sensor IDs must be non-negative and unique");
    sensor_ids.push_back(camera.sensor_id);
    if (camera.name.empty()) fail("camera names must not be empty");
    if (camera.image_topic.empty()) fail("camera." + camera.name + ".image_topic must not be empty");
    if (camera.output_prefix.empty()) fail("camera." + camera.name + ".output_prefix must not be empty");
    if (camera.transport != "compressed" && camera.transport != "raw") {
      fail("camera." + camera.name + ".transport must be either 'raw' or 'compressed'");
    }
  }
}

void ImageInstanceSegmentation::initializeModel() {
  // Pre-populate camera state and client maps before workers start, while still
  // single-threaded, so that later per-camera lookups (from worker threads)
  // only ever call .at() on existing keys and never trigger a concurrent
  // insertion/rehash.
  for (const auto& config : camera_configs_) {
    camera_health_.try_emplace(config.name);
    model_mutexes_.try_emplace(config.name);
    triton_interfaces_.try_emplace(config.name, nullptr);
  }
}

void ImageInstanceSegmentation::initializeModelForCamera(const std::string& camera_name) {
  std::lock_guard<std::mutex> lock(model_mutexes_.at(camera_name));
  auto& state = camera_health_.at(camera_name);
  {
    std::lock_guard<std::mutex> state_lock(state.mutex);
    state.backend_ready = false;
  }
  constexpr auto kRetryDelay = std::chrono::seconds(1);
  while (!stopping_.load() && rclcpp::ok(get_node_base_interface()->get_context())) {
    try {
      // Discover fixed output dimensions using a standard inference before
      // allocating buffers. The public triton_cpp API exposes metadata shapes,
      // which may contain -1.
      const auto output_shape_overrides = probeOutputShapes();

      auto interface = std::make_unique<triton_cpp::TritonInterface>(model_name_, model_version_, server_url_, use_shm_, false,
                                                                     false, triton_client_timeout_s_, false);
      (void)interface->getInputShape(input_tensor_name_);
      for (const auto& name : outputTensorNames()) (void)interface->getOutputShape(name);
      // The model's declared input shape may have dynamic (-1) height/width
      // dims, which the standard (non-variable-size) input path can't size
      // correctly on its own. Pin it to the actual shape we always send so the
      // input buffer is allocated with the right size.
      interface->initInOutputs(output_shape_overrides, std::map<std::string, std::vector<int64_t>>{
                                                           {input_tensor_name_, {1, 3, input_height_, input_width_}}});
      triton_interfaces_.at(camera_name) = std::move(interface);
      {
        std::lock_guard<std::mutex> state_lock(state.mutex);
        state.backend_ready = true;
        state.initialization_error.clear();
      }
      RCLCPP_INFO(get_logger(),
                  "Camera '%s': connected to Triton model '%s:%s' at '%s' "
                  "using input dtype '%s' and output dtype '%s'",
                  camera_name.c_str(), model_name_.c_str(), model_version_.c_str(), server_url_.c_str(),
                  tensorDataTypeName(input_tensor_data_type_).c_str(), tensorDataTypeName(output_tensor_data_type_).c_str());
      return;
    } catch (const std::exception& e) {
      {
        std::lock_guard<std::mutex> state_lock(state.mutex);
        state.initialization_failed = true;
        state.initialization_error = e.what();
      }
      RCLCPP_WARN(get_logger(),
                  "Camera '%s': failed to initialize Triton model '%s:%s' at "
                  "'%s': %s. Retrying in 1s.",
                  camera_name.c_str(), model_name_.c_str(), model_version_.c_str(), server_url_.c_str(), e.what());
      // Component unloading also interrupts retries while the ROS context stays alive.
      const auto retry_at = std::chrono::steady_clock::now() + kRetryDelay;
      while (!stopping_.load() && std::chrono::steady_clock::now() < retry_at &&
             rclcpp::ok(get_node_base_interface()->get_context())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  }
  throw std::runtime_error("Shutdown requested while waiting for Triton connection");
}

void ImageInstanceSegmentation::setupCameras() {
  for (const auto& config : camera_configs_) {
    CameraRuntime runtime;
    runtime.config = config;
    runtime.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    if (tracker_config_.enabled) {
      runtime.tracker = std::make_unique<ByteTracker>(tracker_config_);
    }
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = runtime.callback_group;
    rclcpp::PublisherOptions pub_options;
    pub_options.callback_group = runtime.callback_group;
    runtime.class_mask_publisher = create_publisher<sensor_msgs::msg::Image>(
        joinTopic(config.output_prefix, "segmentation/class_mask"), rclcpp::SensorDataQoS(), pub_options);
    runtime.instance_mask_publisher = create_publisher<sensor_msgs::msg::Image>(
        joinTopic(config.output_prefix, "segmentation/instance_mask"), rclcpp::SensorDataQoS(), pub_options);
    runtime.detections_publisher = create_publisher<vision_msgs::msg::Detection2DArray>(
        joinTopic(config.output_prefix, "segmentation/detections"), rclcpp::SensorDataQoS(), pub_options);
    if (publish_road_users_)
      runtime.road_users_publisher = create_publisher<perception_msgs::msg::ObjectList>(
          joinTopic(config.output_prefix, "perception/road_users"), rclcpp::SensorDataQoS(), pub_options);
    if (publish_traffic_lights_)
      runtime.traffic_lights_publisher = create_publisher<perception_msgs::msg::ObjectList>(
          joinTopic(config.output_prefix, "perception/traffic_lights"), rclcpp::SensorDataQoS(), pub_options);
    if (publish_traffic_signs_)
      runtime.traffic_signs_publisher = create_publisher<perception_msgs::msg::ObjectList>(
          joinTopic(config.output_prefix, "perception/traffic_signs"), rclcpp::SensorDataQoS(), pub_options);
    runtime.image_subscriber = image_transport::create_subscription(
        this, get_node_topics_interface()->resolve_topic_name(config.image_topic),
        [this, name = config.name](const sensor_msgs::msg::Image::ConstSharedPtr& msg) { imageCallback(msg, name); },
        config.transport, rclcpp::SensorDataQoS().get_rmw_qos_profile(), sub_options);
    RCLCPP_INFO(get_logger(),
                "Camera '%s': subscribed to '%s' using '%s' transport, "
                "publishing under '%s'",
                config.name.c_str(), runtime.image_subscriber.getTopic().c_str(), config.transport.c_str(),
                config.output_prefix.c_str());
    cameras_.emplace(config.name, std::move(runtime));
  }
}

cv::Mat ImageInstanceSegmentation::preprocessImage(const cv::Mat& bgr_image, LetterboxInfo& letterbox) const {
  // RF-DETR is trained on images resized without preserving the aspect ratio.
  if (architecture_ == ModelArchitecture::RfDetr) {
    return cv::dnn::blobFromImage(bgr_image, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false,
                                  CV_32F);
  }
  letterbox.scale =
      std::min(static_cast<double>(input_width_) / bgr_image.cols, static_cast<double>(input_height_) / bgr_image.rows);
  letterbox.resized_width = std::clamp(static_cast<int>(std::round(bgr_image.cols * letterbox.scale)), 1, input_width_);
  letterbox.resized_height = std::clamp(static_cast<int>(std::round(bgr_image.rows * letterbox.scale)), 1, input_height_);
  letterbox.pad_x = (input_width_ - letterbox.resized_width) / 2;
  letterbox.pad_y = (input_height_ - letterbox.resized_height) / 2;

  cv::Mat letterboxed(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::Mat resized = letterboxed(cv::Rect(letterbox.pad_x, letterbox.pad_y, letterbox.resized_width, letterbox.resized_height));
  cv::resize(bgr_image, resized, resized.size());
  return cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
}

template <typename TensorScalar>
/** @brief Copy the NCHW float blob into the reusable Triton buffer, converting
   to FP16 when required. */
void copyInputTensor(triton_cpp::TritonInterface& triton_interface,
                     const std::string& tensor_name,
                     int input_height,
                     int input_width,
                     const cv::Mat& blob) {
  const std::size_t element_count =
      static_cast<std::size_t>(1) * 3 * static_cast<std::size_t>(input_height) * static_cast<std::size_t>(input_width);
  const std::size_t byte_size = element_count * sizeof(TensorScalar);
  const auto* input_data = blob.ptr<float>();

  if (triton_interface.usesCudaInputSharedMemory()) {
    if constexpr (std::is_same_v<TensorScalar, float>) {
      triton_interface.copyInputTensorToDevice(tensor_name, input_data, byte_size);
    } else {
      std::vector<TensorScalar> converted(element_count);
      for (std::size_t i = 0; i < element_count; ++i) {
        converted[i] = TensorScalar(input_data[i]);
      }
      triton_interface.copyInputTensorToDevice(tensor_name, converted.data(), byte_size);
    }
    return;
  }
  if constexpr (std::is_same_v<TensorScalar, float>) {
    auto input = triton_interface.getInputTensor<TensorScalar>(tensor_name, 1, 3, input_height, input_width);
    auto& first_elem = input(0, 0, 0, 0);
    auto* dest_ptr = &first_elem;
    std::memcpy(dest_ptr, input_data, byte_size);
  } else {
    auto input = triton_interface.getInputTensor<TensorScalar>(tensor_name, 1, 3, input_height, input_width);
    auto& first_elem = input(0, 0, 0, 0);
    auto* dest_ptr = &first_elem;
    for (std::size_t i = 0; i < element_count; ++i) {
      dest_ptr[i] = TensorScalar(input_data[i]);
    }
  }
}

std::vector<std::string> ImageInstanceSegmentation::outputTensorNames() const {
  if (architecture_ == ModelArchitecture::RfDetr) {
    return {rfdetr_boxes_output_tensor_name_, rfdetr_logits_output_tensor_name_, rfdetr_masks_output_tensor_name_};
  }
  return {yolo_detections_output_tensor_name_, yolo_prototypes_output_tensor_name_};
}

std::map<std::string, std::vector<int64_t>> ImageInstanceSegmentation::probeOutputShapes() {
  // Use the public Triton gRPC API to obtain shapes from an actual result.
  // The configured input dimensions and exported output dimensions must remain
  // fixed.
  namespace tc = triton::client;
  const auto check = [](const tc::Error& error) {
    if (!error.IsOk()) throw std::runtime_error(error.Message());
  };
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  check(tc::InferenceServerGrpcClient::Create(&client, server_url_));
  tc::InferOptions options(model_name_);
  options.model_version_ = model_version_;
  options.client_timeout_ = static_cast<uint64_t>(triton_client_timeout_s_ * 1e6);

  const std::size_t count = static_cast<std::size_t>(3) * input_height_ * input_width_;
  const bool fp16 = input_tensor_data_type_ == TensorDataType::Fp16;
  const std::vector<uint8_t> zeros(count * (fp16 ? sizeof(uint16_t) : sizeof(float)), 0);
  tc::InferInput* raw_input = nullptr;
  check(tc::InferInput::Create(&raw_input, input_tensor_name_, {1, 3, input_height_, input_width_}, fp16 ? "FP16" : "FP32"));
  std::unique_ptr<tc::InferInput> input(raw_input);
  check(input->AppendRaw(zeros.data(), zeros.size()));

  const auto output_names = outputTensorNames();
  std::vector<std::unique_ptr<tc::InferRequestedOutput>> outputs;
  std::vector<const tc::InferRequestedOutput*> raw_outputs;
  for (const auto& name : output_names) {
    tc::InferRequestedOutput* output = nullptr;
    check(tc::InferRequestedOutput::Create(&output, name));
    outputs.emplace_back(output);
    raw_outputs.push_back(output);
  }
  tc::InferResult* raw_result = nullptr;
  check(client->Infer(&raw_result, options, {input.get()}, raw_outputs));
  std::unique_ptr<tc::InferResult> result(raw_result);
  check(result->RequestStatus());
  std::map<std::string, std::vector<int64_t>> shapes;
  for (const auto& name : output_names) {
    check(result->Shape(name, &shapes[name]));
    if (shapes[name].empty() ||
        std::any_of(shapes[name].begin(), shapes[name].end(), [](int64_t dimension) { return dimension <= 0; })) {
      throw std::runtime_error("Model output must have positive, fixed dimensions: " + name);
    }
  }
  return shapes;
}

void ImageInstanceSegmentation::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                                              const std::string& camera_name) {
  const auto input_received_at = std::chrono::steady_clock::now();
  auto& state = camera_health_.at(camera_name);
  bool backend_ready = false;
  const bool valid_stamp = msg->header.stamp.sec >= 0 && msg->header.stamp.nanosec < 1000000000u;
  {
    std::lock_guard<std::mutex> state_lock(state.mutex);
    backend_ready = state.backend_ready;
    if (!backend_ready) ++state.dropped_not_ready;
    if (state.received == 0) state.first_receive = input_received_at;
    state.last_receive = input_received_at;
    state.input_stamp = valid_stamp ? rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type())
                                    : rclcpp::Time(0, 0, get_clock()->get_clock_type());
    ++state.received;
    state.input_valid = msg->width > 0 && msg->height > 0 && !msg->data.empty() && !msg->header.frame_id.empty() && valid_stamp &&
                        state.input_stamp.nanoseconds() > 0;
    if (!state.input_valid) {
      ++state.invalid;
      state.invalid_in_cycle = true;
    }
  }
  if (auto& task = state.topics.at("input").task; task && valid_stamp) task->tick(msg->header.stamp);
  // Receive health includes all frames, but initialization must not queue stale images.
  if (!backend_ready) return;
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  auto& runtime = cameras_.at(camera_name);
  if (stopping_.load()) return;
  int in_flight = runtime.in_flight.load();
  do {
    if (in_flight >= max_in_flight_per_camera_) {
      std::lock_guard<std::mutex> state_lock(state.mutex);
      ++state.dropped;
      return;
    }
  } while (!runtime.in_flight.compare_exchange_weak(in_flight, in_flight + 1));

  try {
    thread_pool_->enqueue([this, msg, camera_name, input_received_at, &runtime] {
      struct ReleaseSlot {
        std::atomic<int>& count;
        ~ReleaseSlot() { count.fetch_sub(1); }
      } release{runtime.in_flight};
      if (stopping_.load()) return;
      try {
        // msg owns the storage behind raw BGR views throughout processing.
        processImage(msg->header, rawImageToBgr(*msg), camera_name, input_received_at);
      } catch (const std::exception& error) {
        recordProcessingFailure(camera_name, error.what());
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Camera '%s': processing failed: %s", camera_name.c_str(),
                             error.what());
      }
    });
  } catch (...) {
    runtime.in_flight.fetch_sub(1);
    throw;
  }
}

void ImageInstanceSegmentation::processImage(const std_msgs::msg::Header& header,
                                             const cv::Mat& bgr_image,
                                             const std::string& camera_name,
                                             std::chrono::steady_clock::time_point input_received_at) {
  auto& camera = cameras_.at(camera_name);
  // A camera's Triton buffers and tracker are shared by its queued requests.
  std::unique_lock<std::mutex> lock(model_mutexes_.at(camera_name));
  if (stopping_.load()) return;
  LetterboxInfo letterbox;
  const cv::Mat blob = preprocessImage(bgr_image, letterbox);
  auto& interface = *triton_interfaces_.at(camera_name);
  auto inference_started_at = std::chrono::steady_clock::now();
  try {
    switch (input_tensor_data_type_) {
      case TensorDataType::Fp32:
        copyInputTensor<float>(interface, input_tensor_name_, input_height_, input_width_, blob);
        break;
      case TensorDataType::Fp16:
        copyInputTensor<Eigen::half>(interface, input_tensor_name_, input_height_, input_width_, blob);
        break;
    }
    inference_started_at = std::chrono::steady_clock::now();
    interface.infer();
  } catch (const std::exception& error) {
    recordProcessingFailure(camera_name, error.what());
    lock.unlock();
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Camera '%s': Triton request failed: %s. Reinitializing.",
                         camera_name.c_str(), error.what());
    initializeModelForCamera(camera_name);
    return;
  }
  const auto inference_finished_at = std::chrono::steady_clock::now();
  std::shared_lock<std::shared_mutex> parameters_lock(runtime_parameters_mutex_);
  const double score_threshold =
      tracker_config_.enabled ? std::min(confidence_threshold_, tracker_config_.low_confidence_threshold) : confidence_threshold_;
  bool invalid_result = false;
  auto detections = decodeDetections(interface, bgr_image.cols, bgr_image.rows, letterbox, score_threshold, invalid_result);
  if (invalid_result) recordProcessingFailure(camera_name, "Non-finite detection candidates in model output");
  if (camera.tracker) detections = camera.tracker->update(detections);
  const auto masks = reconstructMasks(interface, detections, bgr_image.cols, bgr_image.rows, letterbox);
  // Keep publication in the same critical section as inference/tracking.
  publishResults(camera, header, bgr_image, detections, masks);
  auto& state = camera_health_.at(camera_name);
  for (auto& [key, topic] : state.topics) {
    if (key == "input") continue;
    if (auto& task = topic.task; task && header.stamp.sec >= 0 && header.stamp.nanosec < 1000000000u) task->tick(header.stamp);
  }
  {
    std::lock_guard<std::mutex> state_lock(state.mutex);
    const auto finished_at = std::chrono::steady_clock::now();
    state.duration_s = std::chrono::duration<double>(finished_at - input_received_at).count();
    state.peak_duration_s = std::max(state.peak_duration_s, state.duration_s);
    state.processing_failed = invalid_result;
    if (!invalid_result) {
      state.last_success = finished_at;
      ++state.completed;
    }
  }
  if (print_timing_) {
    const auto finished_at = std::chrono::steady_clock::now();
    RCLCPP_INFO(get_logger(),
                "Timing camera '%s': inference=%.3f ms, postprocessing=%.3f "
                "ms, input_to_output=%.3f ms",
                camera_name.c_str(),
                std::chrono::duration<double, std::milli>(inference_finished_at - inference_started_at).count(),
                std::chrono::duration<double, std::milli>(finished_at - inference_finished_at).count(),
                std::chrono::duration<double, std::milli>(finished_at - input_received_at).count());
  }
}

std::vector<Detection> ImageInstanceSegmentation::decodeDetections(triton_cpp::TritonInterface& triton_interface,
                                                                   int image_width,
                                                                   int image_height,
                                                                   const LetterboxInfo& letterbox,
                                                                   double score_threshold,
                                                                   bool& invalid_result) {
  switch (architecture_) {
    case ModelArchitecture::Yolo:
      return applyNms(
          decodeYoloDetections(triton_interface, image_width, image_height, letterbox, score_threshold, invalid_result));
    case ModelArchitecture::Yolo26:
      // YOLO26 end-to-end exports already select their detections without NMS.
      return decodeYoloDetections(triton_interface, image_width, image_height, letterbox, score_threshold, invalid_result);
    case ModelArchitecture::RfDetr:
      return applyNms(decodeRfDetrDetections(triton_interface, image_width, image_height, score_threshold, invalid_result));
  }
  return {};
}

std::vector<Detection> ImageInstanceSegmentation::decodeYoloDetections(triton_cpp::TritonInterface& triton_interface,
                                                                       int image_width,
                                                                       int image_height,
                                                                       const LetterboxInfo& letterbox,
                                                                       double score_threshold,
                                                                       bool& invalid_result) {
  const bool yolo26 = architecture_ == ModelArchitecture::Yolo26;
  const auto runtime_shape = triton_interface.getOutputShape(yolo_detections_output_tensor_name_);
  if (!((runtime_shape.size() == 3 && runtime_shape[0] == 1) || runtime_shape.size() == 2)) {
    throw std::runtime_error(
        "YOLO detection output must have shape [1, "
        "channels, anchors] or [1, anchors, channels]");
  }

  const std::array<int64_t, 3> shape = runtime_shape.size() == 3
                                           ? std::array<int64_t, 3>{runtime_shape[0], runtime_shape[1], runtime_shape[2]}
                                           : std::array<int64_t, 3>{1, runtime_shape[0], runtime_shape[1]};
  const int64_t axis1 = shape[1];
  const int64_t axis2 = shape[2];
  const int num_classes = static_cast<int>(class_names_.size());
  const auto proto_shape = triton_interface.getOutputShape(yolo_prototypes_output_tensor_name_);
  if (!((proto_shape.size() == 4 && proto_shape[0] == 1) || proto_shape.size() == 3)) {
    throw std::runtime_error("YOLO prototype output must have shape [1, masks, height, width]");
  }
  const int64_t mask_count = proto_shape.size() == 4 ? proto_shape[1] : proto_shape[0];
  // YOLO26: x1, y1, x2, y2, sigmoid confidence, class id. YOLOv8/v11: cx, cy, w, h, class scores.
  const int64_t mask_channel_offset = yolo26 ? 6 : 4 + num_classes;
  const int64_t expected_channels = mask_channel_offset + mask_count;
  if ((axis1 == expected_channels) == (axis2 == expected_channels)) {
    throw std::runtime_error(
        "Detection dimensions must identify exactly one "
        "box/class/mask channel axis");
  }
  const bool channels_first = axis1 == expected_channels;
  const int64_t anchors = channels_first ? axis2 : axis1;

  std::vector<Detection> detections;
  auto decode_with_tensor = [&](auto tensor) {
    auto value_at = [&](int64_t channel, int64_t anchor) -> float {
      return static_cast<float>(channels_first ? tensor(0, channel, anchor) : tensor(0, anchor, channel));
    };

    for (int64_t anchor = 0; anchor < anchors; ++anchor) {
      int best_class = 0;
      float best_score = -std::numeric_limits<float>::infinity();
      if (yolo26) {
        best_score = value_at(4, anchor);
        const float class_id = value_at(5, anchor);
        if (!std::isfinite(class_id) || class_id < 0 || class_id >= num_classes || std::floor(class_id) != class_id) {
          invalid_result = true;
          continue;
        }
        best_class = static_cast<int>(class_id);
      } else {
        for (int cls = 0; cls < num_classes; ++cls) {
          const float score = value_at(4 + cls, anchor);
          if (!std::isfinite(score)) invalid_result = true;
          if (score > best_score) {
            best_score = score;
            best_class = cls;
          }
        }
      }
      if (!std::isfinite(best_score)) invalid_result = true;
      if (!std::isfinite(best_score) || best_score < static_cast<float>(score_threshold)) {
        continue;
      }

      const float box_a = value_at(0, anchor);
      const float box_b = value_at(1, anchor);
      const float box_c = value_at(2, anchor);
      const float box_d = value_at(3, anchor);
      if (!std::isfinite(box_a) || !std::isfinite(box_b) || !std::isfinite(box_c) || !std::isfinite(box_d)) {
        invalid_result = true;
        continue;
      }
      const cv::Rect2f model_box = yolo26 ? cv::Rect2f(box_a, box_b, box_c - box_a, box_d - box_b)
                                          : cv::Rect2f(box_a - box_c * 0.5f, box_b - box_d * 0.5f, box_c, box_d);
      cv::Rect2f original_box((model_box.x - letterbox.pad_x) / letterbox.scale,
                              (model_box.y - letterbox.pad_y) / letterbox.scale, model_box.width / letterbox.scale,
                              model_box.height / letterbox.scale);
      original_box = clipBox(original_box, image_width, image_height);
      if (original_box.area() <= 1.0f) {
        continue;
      }

      Detection detection;
      detection.box = original_box;
      detection.class_index = best_class;
      detection.semantic_class_id = static_cast<int>(class_ids_[static_cast<std::size_t>(best_class)]);
      detection.score = best_score;
      detection.mask_coefficients.resize(static_cast<std::size_t>(mask_count));
      bool finite_coefficients = true;
      for (int64_t i = 0; i < mask_count; ++i) {
        const float coefficient = value_at(mask_channel_offset + i, anchor);
        finite_coefficients = finite_coefficients && std::isfinite(coefficient);
        detection.mask_coefficients[static_cast<std::size_t>(i)] = coefficient;
      }
      if (!finite_coefficients) {
        invalid_result = true;
        continue;
      }
      detections.push_back(std::move(detection));
    }
  };

  switch (output_tensor_data_type_) {
    case TensorDataType::Fp32:
      decode_with_tensor(
          triton_interface.getOutputTensor<float>(yolo_detections_output_tensor_name_, shape[0], shape[1], shape[2]));
      break;
    case TensorDataType::Fp16:
      decode_with_tensor(
          triton_interface.getOutputTensor<Eigen::half>(yolo_detections_output_tensor_name_, shape[0], shape[1], shape[2]));
      break;
  }
  return detections;
}

std::vector<Detection> ImageInstanceSegmentation::decodeRfDetrDetections(triton_cpp::TritonInterface& triton_interface,
                                                                         int image_width,
                                                                         int image_height,
                                                                         double score_threshold,
                                                                         bool& invalid_result) {
  const auto boxes_shape = triton_interface.getOutputShape(rfdetr_boxes_output_tensor_name_);
  const auto logits_shape = triton_interface.getOutputShape(rfdetr_logits_output_tensor_name_);
  const int64_t num_classes = static_cast<int64_t>(class_names_.size());
  if (boxes_shape.size() != 3 || boxes_shape[0] != 1 || boxes_shape[2] != 4) {
    throw std::runtime_error("RF-DETR box output must have shape [1, queries, 4]");
  }
  const int64_t queries = boxes_shape[1];
  // RF-DETR exports may append an unused background logit.
  if (logits_shape.size() != 3 || logits_shape[0] != 1 || logits_shape[1] != queries ||
      (logits_shape[2] != num_classes && logits_shape[2] != num_classes + 1)) {
    throw std::runtime_error("RF-DETR logit output must have shape [1, queries, classes] or [1, queries, classes + 1]");
  }

  std::vector<Detection> detections;
  visitTensorDataType(output_tensor_data_type_, [&](auto scalar) {
    using Scalar = decltype(scalar);
    const auto boxes = triton_interface.getOutputTensor<Scalar>(rfdetr_boxes_output_tensor_name_, 1, queries, 4);
    const auto logits = triton_interface.getOutputTensor<Scalar>(rfdetr_logits_output_tensor_name_, 1, queries, logits_shape[2]);
    for (int64_t query = 0; query < queries; ++query) {
      // Boxes are (cx, cy, w, h) normalized to the stretched input, i.e. to the original image.
      const float cx = static_cast<float>(boxes(0, query, 0)) * image_width;
      const float cy = static_cast<float>(boxes(0, query, 1)) * image_height;
      const float width = static_cast<float>(boxes(0, query, 2)) * image_width;
      const float height = static_cast<float>(boxes(0, query, 3)) * image_height;
      if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) || !std::isfinite(height)) {
        invalid_result = true;
        continue;
      }
      const cv::Rect2f box = clipBox({cx - width * 0.5f, cy - height * 0.5f, width, height}, image_width, image_height);
      if (box.area() <= 1.0f) continue;
      for (int64_t cls = 0; cls < num_classes; ++cls) {
        const float logit = static_cast<float>(logits(0, query, cls));
        if (!std::isfinite(logit)) invalid_result = true;
        const float score = 1.0f / (1.0f + std::exp(-logit));
        if (!std::isfinite(score) || score < static_cast<float>(score_threshold)) continue;
        Detection detection;
        detection.box = box;
        detection.class_index = static_cast<int>(cls);
        detection.semantic_class_id = static_cast<int>(class_ids_[static_cast<std::size_t>(cls)]);
        detection.score = score;
        detection.mask_query = static_cast<int>(query);
        detections.push_back(std::move(detection));
      }
    }
  });
  return detections;
}

std::vector<Detection> ImageInstanceSegmentation::applyNms(std::vector<Detection> detections) const {
  std::vector<int> order(detections.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return detections[static_cast<std::size_t>(lhs)].score > detections[static_cast<std::size_t>(rhs)].score;
  });

  std::vector<int> keep;
  std::vector<bool> suppressed(detections.size(), false);
  for (std::size_t i = 0; i < order.size(); ++i) {
    const int candidate = order[i];
    if (suppressed[static_cast<std::size_t>(candidate)]) {
      continue;
    }
    keep.push_back(candidate);
    for (std::size_t j = i + 1; j < order.size(); ++j) {
      const int other = order[j];
      if (suppressed[static_cast<std::size_t>(other)]) {
        continue;
      }
      if (detections[static_cast<std::size_t>(candidate)].class_index !=
          detections[static_cast<std::size_t>(other)].class_index) {
        continue;
      }
      if (intersectionOverUnion(detections[static_cast<std::size_t>(candidate)].box,
                                detections[static_cast<std::size_t>(other)].box) > static_cast<float>(nms_iou_threshold_)) {
        suppressed[static_cast<std::size_t>(other)] = true;
      }
    }
  }
  std::vector<Detection> kept;
  kept.reserve(keep.size());
  for (int index : keep) {
    kept.push_back(std::move(detections[static_cast<std::size_t>(index)]));
  }
  return kept;
}

std::vector<InstanceMask> ImageInstanceSegmentation::reconstructMasks(triton_cpp::TritonInterface& triton_interface,
                                                                      const std::vector<Detection>& detections,
                                                                      int image_width,
                                                                      int image_height,
                                                                      const LetterboxInfo& letterbox) {
  if (detections.empty()) {
    return {};
  }
  if (architecture_ == ModelArchitecture::RfDetr) {
    return reconstructRfDetrMasks(triton_interface, detections, image_width, image_height);
  }
  return reconstructYoloMasks(triton_interface, detections, image_width, image_height, letterbox);
}

std::vector<InstanceMask> ImageInstanceSegmentation::reconstructYoloMasks(triton_cpp::TritonInterface& triton_interface,
                                                                          const std::vector<Detection>& detections,
                                                                          int image_width,
                                                                          int image_height,
                                                                          const LetterboxInfo& letterbox) {
  const auto runtime_shape = triton_interface.getOutputShape(yolo_prototypes_output_tensor_name_);
  if (!((runtime_shape.size() == 4 && runtime_shape[0] == 1) || runtime_shape.size() == 3)) {
    throw std::runtime_error("YOLO prototype output must have shape [1, masks, height, width]");
  }
  const std::array<int64_t, 4> shape =
      runtime_shape.size() == 4 ? std::array<int64_t, 4>{runtime_shape[0], runtime_shape[1], runtime_shape[2], runtime_shape[3]}
                                : std::array<int64_t, 4>{1, runtime_shape[0], runtime_shape[1], runtime_shape[2]};
  const int64_t mask_count = shape[1];
  const int proto_height = static_cast<int>(shape[2]);
  const int proto_width = static_cast<int>(shape[3]);
  RCLCPP_DEBUG(get_logger(), "Prototype output shape: [%ld, %ld, %ld, %ld], mask_count=%ld", shape[0], shape[1], shape[2],
               shape[3], mask_count);

  std::vector<InstanceMask> masks;
  masks.reserve(detections.size());
  const cv::Rect letterbox_roi(letterbox.pad_x, letterbox.pad_y, letterbox.resized_width, letterbox.resized_height);
  const cv::Rect proto_roi(
      static_cast<int>(std::round(letterbox_roi.x * proto_width / static_cast<double>(input_width_))),
      static_cast<int>(std::round(letterbox_roi.y * proto_height / static_cast<double>(input_height_))),
      static_cast<int>(std::round(letterbox_roi.width * proto_width / static_cast<double>(input_width_))),
      static_cast<int>(std::round(letterbox_roi.height * proto_height / static_cast<double>(input_height_))));
  const cv::Rect clipped_proto_roi = proto_roi & cv::Rect(0, 0, proto_width, proto_height);

  const int proto_h = clipped_proto_roi.height;
  const int proto_w = clipped_proto_roi.width;
  if (proto_h <= 0 || proto_w <= 0) {
    throw std::runtime_error("Clipped prototype ROI is empty");
  }
  const int proto_pixels = proto_h * proto_w;
  cv::Mat prototypes_flat(static_cast<int>(mask_count), proto_pixels, CV_32F);
  auto fill_prototypes_flat = [&](auto prototypes) {
    using Scalar = std::remove_cv_t<std::remove_pointer_t<decltype(prototypes.data())>>;
    for (int64_t channel = 0; channel < mask_count; ++channel) {
      cv::Mat source(proto_height, proto_width, cvTypeFor<Scalar>(),
                     const_cast<Scalar*>(prototypes.data()) + channel * proto_height * proto_width);
      cv::Mat destination = prototypes_flat.row(static_cast<int>(channel)).reshape(1, proto_h);
      source(clipped_proto_roi).convertTo(destination, CV_32F);
    }
  };
  switch (output_tensor_data_type_) {
    case TensorDataType::Fp32:
      fill_prototypes_flat(
          triton_interface.getOutputTensor<float>(yolo_prototypes_output_tensor_name_, shape[0], shape[1], shape[2], shape[3]));
      break;
    case TensorDataType::Fp16:
      fill_prototypes_flat(triton_interface.getOutputTensor<Eigen::half>(yolo_prototypes_output_tensor_name_, shape[0], shape[1],
                                                                         shape[2], shape[3]));
      break;
  }

  cv::Mat coefficients(static_cast<int>(detections.size()), static_cast<int>(mask_count), CV_32F);
  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (detections[i].mask_coefficients.size() != static_cast<std::size_t>(mask_count)) {
      throw std::runtime_error("Mask coefficient count does not match prototype channels");
    }
    std::memcpy(coefficients.ptr<float>(static_cast<int>(i)), detections[i].mask_coefficients.data(),
                static_cast<std::size_t>(mask_count) * sizeof(float));
  }
  cv::Mat all_logits;
  cv::gemm(coefficients, prototypes_flat, 1.0, cv::noArray(), 0.0, all_logits);
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const cv::Mat logits = all_logits.row(static_cast<int>(i)).reshape(1, proto_h);
    masks.push_back(maskFromLogits(logits, detections[i].box, image_width, image_height, mask_threshold_));
  }
  return masks;
}

std::vector<InstanceMask> ImageInstanceSegmentation::reconstructRfDetrMasks(triton_cpp::TritonInterface& triton_interface,
                                                                            const std::vector<Detection>& detections,
                                                                            int image_width,
                                                                            int image_height) {
  const auto shape = triton_interface.getOutputShape(rfdetr_masks_output_tensor_name_);
  if (shape.size() != 4 || shape[0] != 1) {
    throw std::runtime_error("RF-DETR mask output must have shape [1, queries, mask_height, mask_width]");
  }
  const int mask_height = static_cast<int>(shape[2]);
  const int mask_width = static_cast<int>(shape[3]);

  std::vector<InstanceMask> masks;
  masks.reserve(detections.size());
  visitTensorDataType(output_tensor_data_type_, [&](auto scalar) {
    using Scalar = decltype(scalar);
    const auto all_logits =
        triton_interface.getOutputTensor<Scalar>(rfdetr_masks_output_tensor_name_, shape[0], shape[1], shape[2], shape[3]);
    cv::Mat logits;
    for (const auto& detection : detections) {
      if (detection.mask_query < 0 || detection.mask_query >= shape[1]) {
        throw std::runtime_error("RF-DETR detection references an invalid mask query");
      }
      // Each query's mask logits cover the whole stretched input, i.e. the original image.
      const cv::Mat source(
          mask_height, mask_width, cvTypeFor<Scalar>(),
          const_cast<Scalar*>(all_logits.data()) + static_cast<std::size_t>(detection.mask_query) * mask_height * mask_width);
      source.convertTo(logits, CV_32F);
      masks.push_back(maskFromLogits(logits, detection.box, image_width, image_height, mask_threshold_));
    }
  });
  return masks;
}

void ImageInstanceSegmentation::publishResults(const CameraRuntime& camera,
                                               const std_msgs::msg::Header& header,
                                               const cv::Mat& bgr_image,
                                               const std::vector<Detection>& detections,
                                               const std::vector<InstanceMask>& masks) const {
  cv::Mat class_mask(bgr_image.rows, bgr_image.cols, CV_16UC1, cv::Scalar(0));
  cv::Mat instance_mask(bgr_image.rows, bgr_image.cols, CV_16UC1, cv::Scalar(0));
  for (std::size_t i = 0; i < detections.size() && i < masks.size(); ++i) {
    if (masks[i].pixels.empty()) continue;
    class_mask(masks[i].roi).setTo(static_cast<std::uint16_t>(detections[i].semantic_class_id), masks[i].pixels);
    const std::uint16_t instance_value = tracker_config_.enabled ? detections[i].track_id : static_cast<std::uint16_t>(i + 1);
    instance_mask(masks[i].roi).setTo(instance_value, masks[i].pixels);
  }

  camera.class_mask_publisher->publish(matToMono16Image(class_mask, header));
  camera.instance_mask_publisher->publish(matToMono16Image(instance_mask, header));
  camera.detections_publisher->publish(makeDetectionsMessage(detections, header));
  publishPerceptionObjects(camera, header, bgr_image, detections, masks);
}

sensor_msgs::msg::Image ImageInstanceSegmentation::matToMono16Image(const cv::Mat& mask,
                                                                    const std_msgs::msg::Header& header) const {
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.height = static_cast<std::uint32_t>(mask.rows);
  msg.width = static_cast<std::uint32_t>(mask.cols);
  msg.encoding = "mono16";
  msg.is_bigendian = false;
  msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(mask.cols * sizeof(std::uint16_t));
  msg.data.resize(static_cast<std::size_t>(msg.step) * msg.height);
  for (int y = 0; y < mask.rows; ++y) {
    const auto* src = mask.ptr<std::uint16_t>(y);
    auto* dst = msg.data.data() + static_cast<std::size_t>(y) * msg.step;
    std::memcpy(dst, src, static_cast<std::size_t>(msg.step));
  }
  return msg;
}

vision_msgs::msg::Detection2DArray ImageInstanceSegmentation::makeDetectionsMessage(const std::vector<Detection>& detections,
                                                                                    const std_msgs::msg::Header& header) const {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(detections.size());
  for (const auto& detection : detections) {
    vision_msgs::msg::Detection2D detection_msg;
    detection_msg.header = header;
    detection_msg.bbox.center.position.x = detection.box.x + detection.box.width * 0.5;
    detection_msg.bbox.center.position.y = detection.box.y + detection.box.height * 0.5;
    detection_msg.bbox.center.theta = 0.0;
    detection_msg.bbox.size_x = detection.box.width;
    detection_msg.bbox.size_y = detection.box.height;
    if (tracker_config_.enabled && detection.track_id != 0) {
      detection_msg.id = std::to_string(detection.track_id);
    }
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = class_names_[static_cast<std::size_t>(detection.class_index)];
    hypothesis.hypothesis.score = detection.score;
    detection_msg.results.push_back(hypothesis);
    msg.detections.push_back(std::move(detection_msg));
  }
  return msg;
}

std::string ImageInstanceSegmentation::joinTopic(const std::string& prefix, const std::string& suffix) {
  if (prefix.empty()) {
    return suffix;
  }
  if (prefix.back() == '/') {
    return prefix + suffix;
  }
  return prefix + "/" + suffix;
}

}  // namespace image_instance_segmentation

RCLCPP_COMPONENTS_REGISTER_NODE(image_instance_segmentation::ImageInstanceSegmentation)
