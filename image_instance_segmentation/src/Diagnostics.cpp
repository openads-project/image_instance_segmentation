// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "image_instance_segmentation/ImageInstanceSegmentation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace image_instance_segmentation {
namespace {
using Status = diagnostic_msgs::msg::DiagnosticStatus;
using Wrapper = diagnostic_updater::DiagnosticStatusWrapper;

/** @brief Capture each topic task's actual result without evaluating it twice. */
class CapturedTopic : public diagnostic_updater::TopicDiagnostic {
 public:
  CapturedTopic(const std::string& topic,
                diagnostic_updater::Updater& updater,
                double& minimum,
                double& maximum,
                double min_age,
                double max_age,
                const rclcpp::Clock::SharedPtr& clock,
                Wrapper& result)
      : TopicDiagnostic(topic,
                        updater,
                        diagnostic_updater::FrequencyStatusParam(&minimum, &maximum, 0.0, 5),
                        diagnostic_updater::TimeStampStatusParam(min_age, max_age),
                        clock),
        result_(result) {}

  /** @brief Evaluate and cache this cycle's topic assessment. */
  void run(Wrapper& status) override {
    TopicDiagnostic::run(status);
    result_ = status;
  }

 private:
  Wrapper& result_;
};

/** @brief Rank known failures above unavailable assessments. */
int severity(unsigned char level) {
  switch (level) {
    case Status::ERROR:
      return 3;
    case Status::STALE:
      return 2;
    case Status::WARN:
      return 1;
    default:
      return 0;
  }
}

/** @brief Render ROS diagnostic levels in check summary values. */
const char* levelName(unsigned char level) {
  switch (level) {
    case Status::ERROR:
      return "ERROR";
    case Status::STALE:
      return "STALE";
    case Status::WARN:
      return "WARN";
    default:
      return "OK";
  }
}
}  // namespace

void ImageInstanceSegmentation::setupDiagnostics() {
  declareAndLoadParameter("diagnostic_updater.use_fqn", diagnostic_use_fqn_,
                          "Prefix diagnostic status names with the fully qualified node name");
  declareAndLoadParameter("diagnostic_updater.period", diagnostic_period_s_, "Diagnostic reporting period in seconds");
  declareAndLoadParameter("diagnostic_updater.inputs.timeout", input_timeout_s_,
                          "Required camera receive timeout in steady seconds");
  declareAndLoadParameter("diagnostic_updater.inputs.max_age", input_max_age_s_, "Maximum input acquisition age in ROS seconds");
  declareAndLoadParameter("diagnostic_updater.inputs.min_age", input_min_age_s_, "Minimum ROS now minus image stamp in seconds");
  processing_timeout_s_ = std::max(5.0, 2.0 * triton_client_timeout_s_);
  declareAndLoadParameter("diagnostic_updater.processing.timeout", processing_timeout_s_,
                          "Maximum time without successful processing in steady seconds");
  declareAndLoadParameter("diagnostic_updater.processing.warn_duration", processing_warn_duration_s_,
                          "Input callback to output duration warning threshold in seconds");
  for (const auto value :
       {diagnostic_period_s_, input_timeout_s_, input_max_age_s_, processing_timeout_s_, processing_warn_duration_s_}) {
    if (!std::isfinite(value) || value <= 0.0)
      throw std::invalid_argument("Diagnostic periods and timeouts must be finite and positive");
  }
  if (!diagnostic_use_fqn_) throw std::invalid_argument("diagnostic_updater.use_fqn must be true");
  if (!std::isfinite(input_min_age_s_) || input_min_age_s_ > input_max_age_s_) {
    throw std::invalid_argument("Diagnostic input timestamp bounds must be finite and ordered");
  }
  if (processing_warn_duration_s_ > processing_timeout_s_) {
    throw std::invalid_argument("Diagnostic processing warn_duration must not exceed timeout");
  }
  if (diagnostic_period_s_ < 0.001) throw std::invalid_argument("Diagnostic reporting period must be at least 0.001 seconds");
  diagnostic_updater_ = std::make_unique<diagnostic_updater::Updater>(this, diagnostic_period_s_, Status::STALE);
  diagnostic_updater_->setHardwareID("none");
  for (auto& [name, state] : camera_health_) {
    const auto& camera = cameras_.at(name);
    const std::map<std::string, std::string> topics{{"input", camera.image_subscriber.getTopic()},
                                                    {"class_mask", camera.class_mask_publisher->get_topic_name()},
                                                    {"instance_mask", camera.instance_mask_publisher->get_topic_name()},
                                                    {"detections", camera.detections_publisher->get_topic_name()}};
    for (const auto& [key, topic] : topics) {
      auto& cfg = state.topics[key];
      const std::string prefix = "diagnostic_updater." + name + "." + key + ".";
      declareAndLoadParameter(prefix + "nominal_frequency", cfg.nominal_frequency,
                              "Expected topic frequency in Hz; zero disables the periodic topic task");
      cfg.min_frequency = 0.8 * cfg.nominal_frequency;
      cfg.max_frequency = 1.2 * cfg.nominal_frequency;
      declareAndLoadParameter(prefix + "min_frequency", cfg.min_frequency, "Minimum topic frequency in Hz");
      declareAndLoadParameter(prefix + "max_frequency", cfg.max_frequency, "Maximum topic frequency in Hz");
      if (!std::isfinite(cfg.nominal_frequency) || !std::isfinite(cfg.min_frequency) || !std::isfinite(cfg.max_frequency) ||
          cfg.nominal_frequency < 0.0 || cfg.min_frequency < 0.0 || cfg.max_frequency < cfg.min_frequency ||
          (cfg.nominal_frequency > 0.0 && cfg.min_frequency <= 0.0) ||
          (cfg.nominal_frequency == 0.0 && (cfg.min_frequency != 0.0 || cfg.max_frequency != 0.0))) {
        throw std::invalid_argument("Invalid frequency bounds for " + prefix);
      }
      if (cfg.nominal_frequency > 0.0) {
        cfg.task = std::make_unique<CapturedTopic>(
            topic, *diagnostic_updater_, cfg.min_frequency, cfg.max_frequency, input_min_age_s_,
            input_max_age_s_ + (key == "input" ? 0.0 : processing_timeout_s_), get_clock(), cfg.result);
      }
    }
  }
  diagnostic_updater_->add("health", this, &ImageInstanceSegmentation::health);
}

void ImageInstanceSegmentation::recordProcessingFailure(const std::string& camera_name, const std::string& message) {
  auto& state = camera_health_.at(camera_name);
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.failed;
  state.processing_failed = true;
  state.failure_in_cycle = true;
  state.processing_error = message;
}

void ImageInstanceSegmentation::health(Wrapper& status) {
  struct Check {
    unsigned char level = Status::OK;
    std::string message = "All checks OK";
  };
  std::map<std::string, Check> checks{{"inputs", {}}, {"processing", {}}, {"data_integrity", {}}};
  Wrapper details;
  const auto steady_now = std::chrono::steady_clock::now();
  const auto ros_now = now();
  const auto assess = [&](const std::string& key, unsigned char level, const std::string& message) {
    auto& check = checks[key];
    if (severity(level) > severity(check.level)) check = {level, message};
  };
  for (auto& [name, state] : camera_health_) {
    CameraHealthSnapshot snapshot;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      snapshot = state;
      state.failure_in_cycle = false;
      state.invalid_in_cycle = false;
      state.peak_duration_s = 0.0;
    }
    const double receive_age =
        std::chrono::duration<double>(steady_now - (snapshot.received ? snapshot.last_receive : diagnostics_started_at_)).count();
    const double output_age =
        std::chrono::duration<double>(steady_now - (snapshot.completed
                                                        ? snapshot.last_success
                                                        : (snapshot.received ? snapshot.first_receive : diagnostics_started_at_)))
            .count();
    const double input_age = snapshot.received ? (ros_now - snapshot.input_stamp).seconds() : 0.0;
    if (receive_age > input_timeout_s_) {
      assess("inputs", Status::ERROR, name + ": required image receive timeout");
    } else if (!snapshot.received) {
      assess("inputs", Status::STALE, name + ": awaiting first image");
    } else if (input_age < input_min_age_s_ || input_age > input_max_age_s_) {
      assess("inputs", Status::ERROR, name + ": image acquisition timestamp outside bounds");
    }
    if (!snapshot.received) assess("data_integrity", Status::STALE, name + ": awaiting first image");
    if (!snapshot.input_valid || snapshot.invalid_in_cycle)
      assess("data_integrity", Status::ERROR, name + ": invalid image dimensions, data, frame or timestamp");
    if (snapshot.failure_in_cycle || snapshot.processing_failed) {
      assess("processing", Status::ERROR, name + ": " + snapshot.processing_error);
    } else if (!snapshot.backend_ready) {
      assess("processing", Status::STALE, name + ": awaiting Triton initialization");
    } else if (snapshot.received && output_age > processing_timeout_s_) {
      assess("processing", Status::ERROR, name + ": no successful result within processing timeout");
    } else if (!snapshot.completed) {
      assess("processing", Status::STALE, name + ": awaiting first result");
    } else if (std::max(snapshot.duration_s, snapshot.peak_duration_s) > processing_warn_duration_s_) {
      assess("processing", Status::WARN, name + ": processing duration above warning threshold");
    }
    if (snapshot.initialization_failed) {
      assess("initialization", snapshot.initialization_error.empty() ? Status::OK : Status::ERROR,
             name + ": " + snapshot.initialization_error);
      details.add("initialization." + name + ".error", snapshot.initialization_error);
    }
    details.add("inputs." + name + ".received_since_start", snapshot.received);
    details.add("inputs." + name + ".receive_age_s", receive_age);
    if (snapshot.received) details.add("inputs." + name + ".acquisition_age_s", input_age);
    details.add("data_integrity." + name + ".invalid_metadata_since_start", snapshot.invalid);
    details.add("processing." + name + ".backend_ready", snapshot.backend_ready);
    details.add("processing." + name + ".completed_since_start", snapshot.completed);
    details.add("processing." + name + ".failed_since_start", snapshot.failed);
    details.add("processing." + name + ".dropped_busy_since_start", snapshot.dropped);
    details.add("processing." + name + ".dropped_not_ready_since_start", snapshot.dropped_not_ready);
    details.add("processing." + name + ".in_flight", cameras_.at(name).in_flight.load());
    details.add("processing." + name + ".output_age_s", output_age);
    details.add("processing." + name + ".duration_s", snapshot.duration_s);
    details.add("processing." + name + ".peak_duration_since_report_s", snapshot.peak_duration_s);
    for (const auto& [key, topic] : state.topics) {
      if (!topic.task) continue;
      const std::string check = "topic." + name + "." + key;
      checks[check] = {topic.result.level,
                       topic.result.message.empty() ? "Frequency and timestamps within bounds" : topic.result.message};
      static const std::map<std::string, std::string> fields{
          {"Events in window", "events_in_window"},
          {"Events since startup", "events_since_start"},
          {"Duration of window (s)", "window_s"},
          {"Actual frequency (Hz)", "frequency_hz"},
          {"Target frequency (Hz)", "target_frequency_hz"},
          {"Minimum acceptable frequency (Hz)", "min_frequency_hz"},
          {"Maximum acceptable frequency (Hz)", "max_frequency_hz"},
          {"Earliest timestamp delay:", "min_delay_s"},
          {"Latest timestamp delay:", "max_delay_s"},
          {"Earliest acceptable timestamp delay:", "min_allowed_delay_s"},
          {"Latest acceptable timestamp delay:", "max_allowed_delay_s"},
          {"Late diagnostic update count:", "late_reports_since_start"},
          {"Early diagnostic update count:", "early_reports_since_start"},
          {"Zero seen diagnostic update count:", "zero_stamp_reports_since_start"}};
      for (const auto& value : topic.result.values) {
        const auto field = fields.find(value.key);
        if (field != fields.end()) details.add(check + "." + field->second, value.value);
      }
    }
  }
  Check worst;
  for (const auto& [key, check] : checks) {
    status.add(key, std::string(levelName(check.level)) + ": " + check.message);
    if (severity(check.level) > severity(worst.level)) worst = {check.level, key + ": " + check.message};
  }
  status.summary(worst.level, worst.message);
  status.values.insert(status.values.end(), details.values.begin(), details.values.end());
}
}  // namespace image_instance_segmentation
