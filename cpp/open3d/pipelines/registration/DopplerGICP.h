// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2021 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
//  Doppler‑GICP: Generalised ICP with scalar Doppler residuals
//  
//  L(R,t) = Σ_i  ||  W_i ( R p_i + t − q_i ) ||²
//         + λ Σ_i ( u_i^{q⊤} R ( s_i^p u_i^p ) − s_i^q )² / σ_v²
//  
//  * W_i  = ( Σ_src + R Σ_tgt Rᵀ )^{-½}  —— Mahalanobis weight from GICP.
//  * u^p, u^q … unit ray directions;   s^p, s^q … scalar Doppler (m/s).


#pragma once

#include <Eigen/Core>
#include <memory>

#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace pipelines {
namespace registration {

class RegistrationResult;

class TransformationEstimationForDopplerGICP
    : public TransformationEstimation {
public:
    ~TransformationEstimationForDopplerGICP() override = default;

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    /// \brief Constructor.
    /// @param lambda_doppler  Weight for Doppler scalar residual (0..1 typical).
    /// @param sigma_v         Doppler noise std (m/s) used to normalize residuals.
    /// @param geometric_kernel Robust kernel applied on whitened geometric residuals.
    /// @param doppler_kernel   Robust kernel applied on Doppler residuals.
    explicit TransformationEstimationForDopplerGICP(
            double lambda_doppler = 0.5,
            double sigma_v = 0.15,
            std::shared_ptr<RobustKernel> geometric_kernel = std::make_shared<L2Loss>(),
            std::shared_ptr<RobustKernel> doppler_kernel   = std::make_shared<L2Loss>())
        : lambda_doppler_(lambda_doppler),
          sigma_v_(sigma_v),
          geometric_kernel_(std::move(geometric_kernel)),
          doppler_kernel_(std::move(doppler_kernel)) {}


public:
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;

    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    // Expose last computed RMSEs (updated by ComputeRMSE)
    mutable double last_geometric_rmse_ = 0.0;   // unitless (whitened)
    mutable double last_doppler_rmse_   = 0.0;   // m/s
    mutable double last_combined_rmse_  = 0.0;   // unitless (matches optimizer cost)
public:
    /// Small constant representing covariance along the normal.
    double epsilon_ = 1e-3;
    // Doppler term settings
    double lambda_doppler_ = 0.5;
    double sigma_v_ = 0.15;  // m/s

    // Robust kernels
    std::shared_ptr<RobustKernel> geometric_kernel_ = std::make_shared<L2Loss>();
    std::shared_ptr<RobustKernel> doppler_kernel_   = std::make_shared<L2Loss>();


private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::DopplerGICP;
    // Context (mutable so we can update from const estimator refs)
    mutable bool has_R_ = false;
    mutable Eigen::Matrix3d current_R_ = Eigen::Matrix3d::Identity();
    mutable const std::vector<Eigen::Vector3d>* src_dirs_ = nullptr;
    mutable const std::vector<Eigen::Vector3d>* tgt_dirs_ = nullptr;

};

// ----------------------------------------------------------------------------
// Doppler-GICP registration driver (mirrors registration_icp pattern)
// ----------------------------------------------------------------------------
// struct DopplerGICPResult {
//     RegistrationResult result;   // ICP result (fitness / rmse / T)
//     // double doppler_rmse = 0.0;   // Optional Doppler RMSE on final corres
// };
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_distance Maximum correspondence points-pair distance.
/// \param init Initial transformation estimation.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]]). \param criteria  Convergence criteria. \param

RegistrationResult RegistrationDopplerGICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimationForDopplerGICP &estimation =
                TransformationEstimationForDopplerGICP(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
