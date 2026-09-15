#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"

namespace open3d {

namespace geometry {
    class PointCloud;
}
namespace pipelines {
namespace registration {

class RegistrationResult;

/// Point-to-Plane ICP with Doppler fusion and dynamic LOS motion prediction.
class TransformationEstimationForDopplerPt2PlaneICP : public TransformationEstimation {
public:
    ~TransformationEstimationForDopplerPt2PlaneICP() override = default;

    TransformationEstimationType GetTransformationEstimationType() const override {
        return type_;
    }

    explicit TransformationEstimationForDopplerPt2PlaneICP(
            double lambda_doppler = 0.01,
            double sigma_v = 0.15, // Doppler noise std (m/s)
            size_t geometric_robust_loss_min_iteration = 0,
            size_t doppler_robust_loss_min_iteration = 2,
            std::shared_ptr<RobustKernel> geometric_kernel = std::make_shared<L2Loss>(),
            std::shared_ptr<RobustKernel> doppler_kernel   = std::make_shared<L2Loss>())
            : lambda_doppler_(lambda_doppler),
            sigma_v_(sigma_v),
            geometric_robust_loss_min_iteration_(geometric_robust_loss_min_iteration),
            doppler_robust_loss_min_iteration_(doppler_robust_loss_min_iteration),
            geometric_kernel_(std::move(geometric_kernel)),
            doppler_kernel_(std::move(doppler_kernel)) {
            if (lambda_doppler_ < 0 || lambda_doppler_ > 1.0) {
            lambda_doppler_ = 0.01;
            }
          }

public:  // API
    // Classic override (not used for this estimator).
    Eigen::Matrix4d ComputeTransformation(const geometry::PointCloud& source,
                                          const geometry::PointCloud& target,
                                          const CorrespondenceSet& corres) const override;

    // Doppler-aware overload (call this from the ICP loop).
    Eigen::Matrix4d ComputeTransformation(const geometry::PointCloud& source,
                                          const geometry::PointCloud& target,
                                          const CorrespondenceSet& corres,
                                          const std::vector<Eigen::Vector3d>& source_dirs,
                                          const std::vector<Eigen::Vector3d>& target_dirs,
                                          const double period,
                                          const Eigen::Matrix4d& transformation,
                                          const size_t iteration) const;

    double ComputeRMSE(const geometry::PointCloud& source,
                       const geometry::PointCloud& target,
                       const CorrespondenceSet& corres) const override;

public:  // visible knobs
    // Optional displacement prior from Doppler ego velocity, in source frame.
    // Zero preserves the paper's two-term objective.
    double ego_translation_weight_ = 0.0;
    Eigen::Vector3d displacement_prior_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d translation_information_ = Eigen::Matrix3d::Identity();
    double lambda_doppler_ = 0.01;  // Doppler weight; geometric weight = 1 - lambda
    double sigma_v_ = 0.15;        // Doppler std (m/s)
    size_t geometric_robust_loss_min_iteration_{0};
    size_t doppler_robust_loss_min_iteration_{2};

    // Robust kernels
    std::shared_ptr<RobustKernel> geometric_kernel_ = std::make_shared<L2Loss>();
    std::shared_ptr<RobustKernel> doppler_kernel_   = std::make_shared<L2Loss>();

    // Optional robust gating for dynamic classification
    double doppler_outlier_threshold_ = 10.0; // fallback gate if sigma_v_ <= 0 or k < 0
    double beta_perp_ = 0.0;                      // weight for P-term (0 disables)
    double k_dynamic_gate_ = 3.0;                 // gate multiplier on sigma_v
    bool enable_dynamic_compensation_ {false}; // whether to use dynamic compensation

    // Iteration thresholds for enabling robust kernels (optional)
    // Eigen::Vector3d v_sensor_src_ = Eigen::Vector3d::Zero(); // ego-vel in source frame
    // Eigen::Vector3d v_sensor_tgt_ = Eigen::Vector3d::Zero(); // ego-vel in target frame
private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::DopplerPt2PlaneICP;
};

/// Registration wrapper.
RegistrationResult RegistrationDopplerPt2PlaneICP(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const std::vector<Eigen::Vector3d>& source_dirs,
        const std::vector<Eigen::Vector3d>& target_dirs,
        double max_correspondence_distance,
        const Eigen::Matrix4d& init,
        const TransformationEstimationForDopplerPt2PlaneICP& estimation,
        const ICPConvergenceCriteria& criteria,
        const double period = 0.1F);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
