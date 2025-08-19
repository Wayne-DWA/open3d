// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/DopplerPt2PlaneICP.h"

#include <Eigen/Dense>
#include <iostream>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Vector6d;

Eigen::Matrix4d TransformationEstimationForDopplerPt2PlaneICP::ComputeTransformation(
        const geometry::PointCloud& /*source*/,
        const geometry::PointCloud& /*target*/,
        const CorrespondenceSet& /*corres*/) const {
    utility::LogError(
        "Use the Doppler-aware overload of ComputeTransformation with LOS dirs and current transform.");
    return Matrix4d::Identity();
}

// Point-to-Plane + Doppler + Dynamic prediction
Eigen::Matrix4d TransformationEstimationForDopplerPt2PlaneICP::ComputeTransformation(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const CorrespondenceSet& corres,
        const std::vector<Vector3d>& source_dirs,
        const std::vector<Vector3d>& target_dirs,
        const double period,
        const Matrix4d& transformation,
        const size_t iteration) const {
    if (corres.empty()) {
        utility::LogError("No correspondences found between source and target pointcloud.");
    }
    if (!target.HasNormals()) {
        utility::LogError("Doppler Pt2Plane ICP requires target normals.");
    }
    if (!source.HasDopplers()) {
        utility::LogError("Doppler Pt2Plane ICP requires source dopplers.");
    }
    if (source_dirs.size() != source.points_.size() ||
        target_dirs.size() != target.points_.size()) {
        utility::LogError("LOS direction arrays must match point counts.");
    }

    const double lambda_g = 1.0 - lambda_doppler_;
    const double sqrt_lambda_v = std::sqrt(lambda_doppler_);
    const double sqrt_lambda_g = std::sqrt(lambda_g);

    // Small-angle state (not explicitly used here, kept for completeness)
    (void)transformation; // current T, if needed

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Vector6d, utility::Vector6d_allocator>& J_r,
                std::vector<double>& r,
                std::vector<double>& w) {
                const size_t cs = corres[i][0];
                const size_t ct = corres[i][1];

                const Vector3d& ps = source.points_[cs];   // in current frame (pcd already transformed)
                const Vector3d& pt = target.points_[ct];
                const Vector3d& nt = target.normals_[ct];  // plane normal at target

                const Vector3d& up = source_dirs[cs];       // LOS dir at source point
                const Vector3d& uq = target_dirs[ct];       // LOS dir at target point
                const double sp = source.dopplers_[cs];     // m/s
                const double sq = target.dopplers_.empty() ? 0.0 : target.dopplers_[ct];

                // Geometric residual: point-to-plane
                // Static: r_g = n^T (ps - pt)
                // Dynamic: r_g = n^T (ps - pt + alpha_p u_p + alpha_q u_q)
                const Vector3d d = ps - pt;
                // Doppler scalar residual (small-angle; R ≈ I inside one iteration)
                const Vector3d Rvp = sp * up; // LOS velocity mapped to target frame
                const double rv = uq.dot(Rvp) - sq; // m/s
                const Vector3d Jv_rot_vec = Rvp.cross(uq); // d/d(delta_theta)
                const int n_rows = 2;
                J_r.resize(n_rows);
                r.resize(n_rows);
                w.resize(n_rows);
                // Geometric Jacobian, static or dynamic
                // r_g = sqrt_lambda_g * n^T d_dyn
                const double rg = sqrt_lambda_g * nt.dot(d);
                // J_g wrt rotation using cross: d/dδθ [ n^T (δθ × ps + α_p δθ × u_p) ] = (ps × n + α_p u_p × n)^T
                Vector3d rot_geo = ps.cross(nt);
                // Compose 6-d row Jacobian (rotation first, then translation)
                Vector6d Jg = Vector6d::Zero();

                Jg.head<3>() = sqrt_lambda_g * rot_geo;
                Jg.tail<3>() = sqrt_lambda_g * nt;       // d/dt: n^T * I
                J_r[0] = Jg;
                r[0]   = rg;
                w[0]   = (iteration >= geometric_robust_loss_min_iteration_) ?
                            geometric_kernel_->Weight(r[0]) : 1.0;

                // Doppler scalar residual (normalized) and Jacobian
                const double rv_n = (sigma_v_ > 0.0 ? rv / sigma_v_ : rv);
                r[1] = sqrt_lambda_v * rv_n;
                w[1] = (iteration >= doppler_robust_loss_min_iteration_) ?
                            doppler_kernel_->Weight(r[1]) : 1.0;
                Vector6d Jv = Vector6d::Zero();
                Jv.head<3>() = sqrt_lambda_v * (sigma_v_ > 0.0 ? (Jv_rot_vec / sigma_v_) : Jv_rot_vec);
                // no translation effect
                J_r[1] = Jv;

                // Ego-motion compensation for Doppler gating and dynamic prediction
                // const double sp_comp = sp + up.dot(v_sensor_src_);
                // const double sq_comp = sq + uq.dot(v_sensor_tgt_);

                // Doppler scalar residual (small-angle; R ≈ I inside one iteration)
                // const Vector3d Rvp = sp_comp * up; // LOS velocity mapped to target frame
                // const double rv = uq.dot(Rvp) - sq_comp; // m/s
                // const Vector3d Jv_rot_vec = Rvp.cross(uq); // d/d(delta_theta)

                // Dynamic gating
                // const double gate = (k_dynamic_gate_ > 0.0 && sigma_v_ > 0.0)
                //                         ? (k_dynamic_gate_ * sigma_v_)
                //                         : doppler_outlier_threshold_;
                // const bool is_dynamic = enable_dynamic_compensation_ && (std::abs(rv) > gate);

                // // Reserve rows: 1 (geom) + 1 (doppler) + optional 3 (perp)
                // const bool use_perp = (beta_perp_ > 0.0);
                // const int n_rows = use_perp ? 5 : 2;
                // J_r.resize(n_rows);
                // r.resize(n_rows);
                // w.resize(n_rows);

                // // Geometric Jacobian, static or dynamic
                // Vector3d d_dyn = d;
                // if (is_dynamic) {
                //     const double alpha_p = sp_comp * period;
                //     const double alpha_q = sq_comp * period;
                //     d_dyn = d + alpha_p * up + alpha_q * uq;
                // }
                // // r_g = sqrt_lambda_g * n^T d_dyn
                // const double rg = sqrt_lambda_g * nt.dot(d_dyn);
                // // J_g wrt rotation: sqrt_lambda_g * n^T ( -[ps]_x - alpha_p [u_p]_x )
                // Vector3d rot_geo = ps.cross(nt); // because n^T (-[ps]_x) = ( -[ps]_x^T n )^T; we store row later
                // if (is_dynamic) {
                //     const double alpha_p = sp_comp * period;
                //     rot_geo += alpha_p * up.cross(nt);
                // }
                // // Compose 6-d row Jacobian (rotation first, then translation)
                // Vector6d Jg = Vector6d::Zero();
                // Jg.head<3>() = sqrt_lambda_g * rot_geo; // note: we treat row via vector; Open3D expects per-row packing
                // Jg.tail<3>() = sqrt_lambda_g * nt;       // d/dt: n^T * I
                // J_r[0] = Jg;
                // r[0]   = rg;
                // w[0]   = (iteration >= geometric_robust_loss_min_iteration_) ?
                //             geometric_kernel_->Weight(r[0]) : 1.0;

                // // Doppler scalar residual (normalized) and Jacobian
                // const double rv_n = (sigma_v_ > 0.0 ? rv / sigma_v_ : rv);
                // r[1] = sqrt_lambda_v * rv_n;
                // w[1] = (iteration >= doppler_robust_loss_min_iteration_) ?
                //             doppler_kernel_->Weight(r[1]) : 1.0;
                // Vector6d Jv = Vector6d::Zero();
                // Jv.head<3>() = sqrt_lambda_v * (sigma_v_ > 0.0 ? (Jv_rot_vec / sigma_v_) : Jv_rot_vec);
                // // no translation effect
                // J_r[1] = Jv;

                // if (use_perp) {
                //     // Ray-perpendicular geometric consistency
                //     // r_perp = sqrt(beta) * P * d_dyn, where P = I - u_q u_q^T
                //     const Matrix3d P = Matrix3d::Identity() - uq * uq.transpose();
                //     const Vector3d rP = P * d_dyn;
                //     const double sqrt_beta = std::sqrt(beta_perp_);
                //     // Build rotation rows using cross only:
                //     // For each component i, row rotation part a_i^T = sqrt_beta * ( ps × v + α_p u_p × v )^T, where v = P.col(i)
                //     // Translation part is sqrt_beta * P.row(i)
                //     const double alpha_p = is_dynamic ? (sp_comp * period) : 0.0;
                //     const Vector3d m = ps + alpha_p * up;
                //     for (int k = 0; k < 3; ++k) {
                //         const Vector3d v = P.col(k);
                //         Vector6d row = Vector6d::Zero();
                //         // rotation (1x3): (ps × v + α_p u_p × v)^T
                //         Vector3d a_rot = ps.cross(v);
                //         if (alpha_p != 0.0) a_rot += alpha_p * up.cross(v);
                //         row.head<3>() = sqrt_beta * a_rot;
                //         // translation (1x3): P.row(k)
                //         row.tail<3>() = sqrt_beta * P.row(k).transpose();
                //         J_r[2 + k] = row;
                //         r[2 + k]   = sqrt_beta * rP(k);
                //         w[2 + k]   = geometric_kernel_ ? geometric_kernel_->Weight(r[2 + k]) : 1.0;
                //     }
                // }
            };

    Eigen::Matrix<double, 6, 6> JTJ;
    Vector6d JTr;
    double r2 = -1.0;
    std::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix<double,6,6>, Eigen::Vector6d>(
                    compute_jacobian_and_residual, static_cast<int>(corres.size()));

    bool is_success = false;
    Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Matrix4d::Identity();
}

// Standard RMSE: geometric point-to-plane only (for Open3D metrics)
double TransformationEstimationForDopplerPt2PlaneICP::ComputeRMSE(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const CorrespondenceSet& corres) const {
    if (corres.empty() || !target.HasNormals()) return 0.0;
    double err = 0.0, rg;
    for (const auto& c : corres) {
        const Vector3d& ps = source.points_[c[0]];
        const Vector3d& pt = target.points_[c[1]];
        const Vector3d& nt = target.normals_[c[1]];
        rg = (ps - pt).dot(nt);
        err += rg * rg;
    }
    return std::sqrt(err / static_cast<double>(corres.size()));
}

RegistrationResult RegistrationDopplerPt2PlaneICP(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const std::vector<Vector3d>& source_dirs,
        const std::vector<Vector3d>& target_dirs,
        double max_correspondence_distance,
        const Matrix4d& init,
        const TransformationEstimationForDopplerPt2PlaneICP& estimation,
        const ICPConvergenceCriteria& criteria,
        const double period) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }
    if (!target.HasNormals()) {
        utility::LogError("TransformationEstimationForDopplerPt2PlaneICP requires target normals.");
    }
    if (!source.HasDopplers()) {
        utility::LogError("TransformationEstimationForDopplerPt2PlaneICP requires source dopplers.");
    }

    Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }

    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);

    int i;
    bool converged = false;

    for (i = 0; i < criteria.max_iteration_; ++i) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);

        // Update ego velocity from last update translation (target frame)

        // Compute the transform update with Doppler + dynamic prediction
        Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_,
                source_dirs, target_dirs, period, transformation, i);
        transformation = update * transformation;
        pcd.Transform(update);
        // Update registration result
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);

        if (std::abs(backup.fitness_ - result.fitness_) < criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) < criteria.relative_rmse_) {
            converged = true;
            break;
        }
    }

    result.num_iterations_ = i;
    result.converged_ = converged;
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d