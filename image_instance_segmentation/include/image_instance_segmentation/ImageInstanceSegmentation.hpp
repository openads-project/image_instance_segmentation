// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <triton_cpp/triton_interface.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <diagnostic_updater/publisher.hpp>
#include <functional>
#include <image_transport/image_transport.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <perception_msgs/msg/object_list.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <shared_mutex>
#include <std_msgs/msg/header.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace image_instance_segmentation {

struct CameraConfig {
  std::string name;
  int64_t sensor_id = 1;
  std::string image_topic = "~/image";
  std::string transport = "raw";
  std::string output_prefix = "~";
};

struct LetterboxInfo {
  double scale = 1.0;
  int pad_x = 0;
  int pad_y = 0;
  int resized_width = 0;
  int resized_height = 0;
};

struct Detection {
  cv::Rect2f box;
  int class_index = 0;
  int semantic_class_id = 0;
  float score = 0.0f;
  std::uint16_t track_id = 0;
  std::uint64_t object_id = 0;
  bool is_tracked = false;
  std::vector<float> mask_coefficients;  // YOLO: coefficients for the prototype masks
  int mask_query = -1;                   // RF-DETR: query whose mask logits belong to this detection
};

// Keep each instance mask cropped to its detection box to avoid an image-sized
// allocation and repeated full-image scans for every detected object.
struct InstanceMask {
  cv::Rect roi;
  cv::Mat pixels;
};

struct TrackerConfig {
  bool enabled = true;
  std::string backend = "bytetrack";
  double high_confidence_threshold = 0.5;
  double low_confidence_threshold = 0.1;
  double new_track_confidence_threshold = 0.6;
  double match_iou_threshold = 0.3;
  int max_lost_frames = 30;
  double min_box_area_px = 25.0;
};

enum class TensorDataType {
  Fp32,
  Fp16,
};

enum class ModelArchitecture {
  Yolo,    // YOLOv8/v11: letterbox, per-anchor class scores and mask coefficients, NMS
  Yolo26,  // YOLO26 end-to-end: letterbox, selected detections, no NMS
  RfDetr,  // RF-DETR-Seg: stretch, per-query class logits and mask logits, NMS
};

struct PerceptionClass {
  std::uint8_t model_id = 0;
  std::uint8_t classification = 0;
};

class ITracker;
class ThreadPool;

class ImageInstanceSegmentation : public rclcpp::Node {
 public:
  /** @brief Construct camera workers and Triton clients from the node options.
   */
  explicit ImageInstanceSegmentation(const rclcpp::NodeOptions& options);
  /** @brief Stop admission, unsubscribe cameras, and join workers before
   * releasing camera state. */
  ~ImageInstanceSegmentation() override;

 private:
  struct TopicHealth {
    double nominal_frequency = 0.0;
    double min_frequency = 0.0;
    double max_frequency = 0.0;
    diagnostic_updater::DiagnosticStatusWrapper result;
    std::unique_ptr<diagnostic_updater::TopicDiagnostic> task;
  };

  struct CameraHealthSnapshot {
    std::chrono::steady_clock::time_point first_receive{};
    std::chrono::steady_clock::time_point last_receive{};
    std::chrono::steady_clock::time_point last_success{};
    rclcpp::Time input_stamp{0, 0, RCL_ROS_TIME};
    std::uint64_t received = 0;
    std::uint64_t dropped = 0;
    std::uint64_t dropped_not_ready = 0;
    std::uint64_t completed = 0;
    std::uint64_t failed = 0;
    std::uint64_t invalid = 0;
    bool backend_ready = false;
    bool initialization_failed = false;
    bool input_valid = true;
    bool invalid_in_cycle = false;
    bool processing_failed = false;
    bool failure_in_cycle = false;
    double duration_s = 0.0;
    double peak_duration_s = 0.0;
    std::string initialization_error;
    std::string processing_error;
  };

  struct CameraHealth : CameraHealthSnapshot {
    std::mutex mutex;
    std::map<std::string, TopicHealth> topics;
  };

  /** @brief Configure topic tasks and the single overall health assessment. */
  void setupDiagnostics();
  /** @brief Summarize topic results and camera/backend snapshots once per cycle. */
  void health(diagnostic_updater::DiagnosticStatusWrapper& status);
  /** @brief Record a failed frame without publishing diagnostics from a worker. */
  void recordProcessingFailure(const std::string& camera_name, const std::string& message);

  struct CameraRuntime {
    CameraConfig config;
    image_transport::Subscriber image_subscriber;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr class_mask_publisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr instance_mask_publisher;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_publisher;
    rclcpp::Publisher<perception_msgs::msg::ObjectList>::SharedPtr road_users_publisher;
    rclcpp::Publisher<perception_msgs::msg::ObjectList>::SharedPtr traffic_lights_publisher;
    rclcpp::Publisher<perception_msgs::msg::ObjectList>::SharedPtr traffic_signs_publisher;
    rclcpp::CallbackGroup::SharedPtr callback_group;
    std::unique_ptr<ITracker> tracker;
    std::atomic<int> in_flight{0};

    /** @brief Create empty runtime state with no in-flight frames. */
    CameraRuntime();
    /** @brief Release the tracker in the implementation that defines it. */
    ~CameraRuntime();
    /** @brief Prevent copying subscriptions, tracker ownership, and admission
     * counters. */
    CameraRuntime(const CameraRuntime&) = delete;
    /** @brief Prevent copy assignment of camera runtime ownership. */
    CameraRuntime& operator=(const CameraRuntime&) = delete;

    /** @brief Transfer camera resources and snapshot the atomic in-flight
     * count. */
    CameraRuntime(CameraRuntime&& other) noexcept;
    /** @brief Move camera resources and copy the atomic in-flight count. */
    CameraRuntime& operator=(CameraRuntime&& other) noexcept;
  };

  /** @brief Declare a parameter with its current default and load any
   * launch/YAML override.
   * @param name Fully qualified ROS parameter name.
   * @param value Default on entry, effective value on return.
   * @param description Human-readable description exposed by ROS parameter
   * introspection.
   * @param add_to_auto_reconfigurable_params Apply successful runtime updates to the value.
   * @param is_required Require an explicit parameter override at startup.
   * @param read_only Prevent changes after declaration.
   * @param from_value Optional inclusive lower bound.
   * @param to_value Optional inclusive upper bound.
   * @param step_value Optional numeric step; zero permits continuous values.
   * @param additional_constraints Constraints exposed through parameter introspection.
   */
  template <typename T>
  void declareAndLoadParameter(const std::string& name,
                               T& value,
                               const std::string& description,
                               bool add_to_auto_reconfigurable_params = false,
                               bool is_required = false,
                               bool read_only = true,
                               const std::optional<double>& from_value = std::nullopt,
                               const std::optional<double>& to_value = std::nullopt,
                               const std::optional<double>& step_value = std::nullopt,
                               const std::string& additional_constraints = "");
  /** @brief Validate runtime updates without changing active processing settings. */
  rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter>& parameters) const;
  /** @brief Apply a committed parameter update between postprocessing operations. */
  void applyParameters(const std::vector<rclcpp::Parameter>& parameters);
  /** @brief Declare and load global parameters and each configured camera. */
  void loadParameters();
  /** @brief Reject invalid dimensions, class mappings, thresholds, and camera
   * settings. */
  void validateParameters() const;
  /** @brief Allocate stable per-camera state before starting model workers. */
  void initializeModel();
  /** @brief Initialize one camera client, resolving tensor shapes before
   * allocating buffers. */
  void initializeModelForCamera(const std::string& camera_name);
  /** @brief Output tensors read by the configured architecture. */
  std::vector<std::string> outputTensorNames() const;
  /** @brief Run a zero-filled inference to resolve dynamic output shapes at
   * startup. */
  std::map<std::string, std::vector<int64_t>> probeOutputShapes();
  /** @brief Create per-camera subscriptions, output publishers, callback
   * groups, and trackers. */
  void setupCameras();
  /** @brief Admit a frame within the per-camera limit and queue work while
   * retaining its message storage. */
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg, const std::string& camera_name);
  /** @brief Preprocess, infer, decode, track, reconstruct masks, and publish
   * one admitted frame. */
  void processImage(const std_msgs::msg::Header& header,
                    const cv::Mat& bgr_image,
                    const std::string& camera_name,
                    std::chrono::steady_clock::time_point input_received_at);

  /** @brief Letterbox (YOLO) or stretch (RF-DETR) the image into a contiguous
   * RGB NCHW float blob normalized to [0, 1]. */
  cv::Mat preprocessImage(const cv::Mat& bgr_image, LetterboxInfo& letterbox) const;
  /** @brief Decode the configured architecture's outputs into original-image
   * detections. */
  std::vector<Detection> decodeDetections(triton_cpp::TritonInterface& triton_interface,
                                          int image_width,
                                          int image_height,
                                          const LetterboxInfo& letterbox,
                                          double score_threshold,
                                          bool& invalid_result);
  /** @brief Decode validated YOLO tensor layouts into original-image boxes and
   * mask coefficients. */
  std::vector<Detection> decodeYoloDetections(triton_cpp::TritonInterface& triton_interface,
                                              int image_width,
                                              int image_height,
                                              const LetterboxInfo& letterbox,
                                              double score_threshold,
                                              bool& invalid_result);
  /** @brief Decode RF-DETR boxes and per-class logits into one detection per
   * query and class above the threshold. */
  std::vector<Detection> decodeRfDetrDetections(triton_cpp::TritonInterface& triton_interface,
                                                int image_width,
                                                int image_height,
                                                double score_threshold,
                                                bool& invalid_result);
  /** @brief Keep confidence-ordered detections using class-specific IoU
   * suppression. */
  std::vector<Detection> applyNms(std::vector<Detection> detections) const;
  /** @brief Publish masks, detection metadata, and enabled perception object lists. */
  void publishResults(const CameraRuntime& camera,
                      const std_msgs::msg::Header& header,
                      const cv::Mat& bgr_image,
                      const std::vector<Detection>& detections,
                      const std::vector<InstanceMask>& masks) const;
  /** @brief Reconstruct each detection's mask within its box ROI. */
  std::vector<InstanceMask> reconstructMasks(triton_cpp::TritonInterface& triton_interface,
                                             const std::vector<Detection>& detections,
                                             int image_width,
                                             int image_height,
                                             const LetterboxInfo& letterbox);
  /** @brief Multiply YOLO mask coefficients by the prototypes of the letterbox
   * content. */
  std::vector<InstanceMask> reconstructYoloMasks(triton_cpp::TritonInterface& triton_interface,
                                                 const std::vector<Detection>& detections,
                                                 int image_width,
                                                 int image_height,
                                                 const LetterboxInfo& letterbox);
  /** @brief Take each detection's RF-DETR query mask logits. */
  std::vector<InstanceMask> reconstructRfDetrMasks(triton_cpp::TritonInterface& triton_interface,
                                                   const std::vector<Detection>& detections,
                                                   int image_width,
                                                   int image_height);
  /** @brief Resolve model class names to perception state models and road-user classes. */
  void configurePerceptionClasses();
  /** @brief Publish enabled image-only object lists, retaining optional ROI pixels. */
  void publishPerceptionObjects(const CameraRuntime& camera,
                                const std_msgs::msg::Header& header,
                                const cv::Mat& bgr_image,
                                const std::vector<Detection>& detections,
                                const std::vector<InstanceMask>& masks) const;
  /** @brief Copy a mono16 mask into a ROS image while preserving the input
   * header. */
  sensor_msgs::msg::Image matToMono16Image(const cv::Mat& mask, const std_msgs::msg::Header& header) const;
  /** @brief Build original-image detections with class scores and optional
   * per-camera track IDs. */
  vision_msgs::msg::Detection2DArray makeDetectionsMessage(const std::vector<Detection>& detections,
                                                           const std_msgs::msg::Header& header) const;
  /** @brief Join a configured topic prefix and suffix with exactly one
   * separator. */
  static std::string joinTopic(const std::string& prefix, const std::string& suffix);

  std::map<std::string, CameraHealth> camera_health_;
  std::chrono::steady_clock::time_point diagnostics_started_at_ = std::chrono::steady_clock::now();
  double diagnostic_period_s_ = 1.0;
  bool diagnostic_use_fqn_ = true;
  double input_timeout_s_ = 2.0;
  double input_max_age_s_ = 1.0;
  double input_min_age_s_ = -0.05;
  double processing_timeout_s_ = 5.0;
  double processing_warn_duration_s_ = 0.1;

  std::string server_url_ = "triton-server:8001";
  std::string model_name_ = "rfdetr-seg-small";
  std::string model_version_ = "1";
  double triton_client_timeout_s_ = 2.0;
  bool use_shm_ = false;

  std::string architecture_name_ = "rfdetr";
  ModelArchitecture architecture_ = ModelArchitecture::RfDetr;
  int input_width_ = 1280;
  int input_height_ = 720;
  std::string input_tensor_name_ = "input";
  std::string yolo_detections_output_tensor_name_ = "output0";
  std::string yolo_prototypes_output_tensor_name_ = "output1";
  std::string rfdetr_boxes_output_tensor_name_ = "dets";
  std::string rfdetr_logits_output_tensor_name_ = "labels";
  std::string rfdetr_masks_output_tensor_name_ = "masks";
  TensorDataType input_tensor_data_type_ = TensorDataType::Fp16;
  TensorDataType output_tensor_data_type_ = TensorDataType::Fp16;
  std::string input_dtype_ = "fp16";
  std::string output_dtype_ = "fp16";
  std::vector<std::string> camera_names_ = {"camera"};
  // Keep this default on one line for the generated parameter table.
  // clang-format off
  std::vector<std::string> class_names_ = {"PEDESTRIAN", "BICYCLE", "MOTORCYCLE", "CAR", "UTILITY", "BUS", "ANIMAL", "VRU", "MICRO", "TRAFFIC_LIGHT", "TRAFFIC_SIGN"};
  // clang-format on
  std::vector<int64_t> class_ids_ = {1, 2, 3, 4, 5, 7, 8, 12, 13, 20, 29};

  bool publish_road_users_ = false;
  bool publish_traffic_lights_ = false;
  bool publish_traffic_signs_ = false;
  bool perception_include_crops_ = true;
  bool perception_include_masks_ = true;
  std::vector<PerceptionClass> perception_classes_;
  mutable std::atomic<std::uint64_t> next_untracked_object_id_{1};

  double confidence_threshold_ = 0.25;
  double mask_threshold_ = 0.5;
  double nms_iou_threshold_ = 0.45;
  TrackerConfig tracker_config_;
  bool print_timing_ = false;

  std::vector<CameraConfig> camera_configs_;
  std::unordered_map<std::string, CameraRuntime> cameras_;

  int max_in_flight_per_camera_ = 1;

  // Each camera has an independent Triton client and mutex. The mutex protects
  // its reusable buffers and tracker while different cameras run concurrently.
  // Both maps are populated before workers start and remain unchanged at runtime.
  std::unordered_map<std::string, std::mutex> model_mutexes_;
  std::unordered_map<std::string, std::unique_ptr<triton_cpp::TritonInterface>> triton_interfaces_;

  // Processing takes a shared lock only after inference; parameter commits take
  // an exclusive lock so a frame uses one consistent set of runtime settings.
  mutable std::shared_mutex runtime_parameters_mutex_;
  std::unordered_map<std::string, std::function<void(const rclcpp::Parameter&)>> auto_reconfigurable_params_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;
  rclcpp::node_interfaces::PostSetParametersCallbackHandle::SharedPtr parameters_post_callback_handle_;

  // Workers are joined before camera state and Triton clients are released.
  std::mutex admission_mutex_;
  std::atomic<bool> stopping_{false};
  std::unique_ptr<ThreadPool> thread_pool_;
  // Destroy the updater timer before the tasks and snapshots it references.
  std::unique_ptr<diagnostic_updater::Updater> diagnostic_updater_;
};

}  // namespace image_instance_segmentation
