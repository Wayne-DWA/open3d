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

#include "open3d/pipelines/registration/DopplerVelocityICP.h"

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
// This estimator adds a velocity vector consistency term:
//   rv_i = R * v_i^p - v_i^q
// Jacobians (left-multiplicative update):
//   geometric: Jg = [ -[p'_i]_x | I_3 ] with p'_i in target frame
//   velocity : Jv = [ -[R v_i^p]_x |  0  ] (translation-free)

Eigen::Matrix4d TransformationEstimationForDopplerVelocityICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    utility::LogError(
            "This method should not be called for DopplerVelocityICP."
            "Please call the overload that provides velocity vectors.");
    return Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d TransformationEstimationForDopplerVelocityICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<Eigen::Vector3d> &source_directions,
        const std::vector<Eigen::Vector3d> &target_directions,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &transformation,
        const size_t iteration) const {
    if (corres.empty()) {
        utility::LogError(
                "No correspondences found between source and target "
                "pointcloud.");
    }
    // point to point ICP does not require normals
//     if (!target.HasNormals()) {
//         utility::LogError(
//                 "DopplerVelocityICP requires target pointcloud to have normals.");
//     }
    if (!source.HasDopplers()) {
        utility::LogError(
                "DopplerVelocityICP requires source pointcloud to have Dopplers.");
    }
    // we don't use period in this method
//     if (std::abs(period) < 1e-3) {
//         utility::LogError("Time period too small.");
//     }

    const double lambda_geometric = 1.0 - lambda_doppler_;
    const double sqrt_lambda_doppler = std::sqrt(lambda_doppler_);
    const double sqrt_lambda_geometric = std::sqrt(lambda_geometric);

//     const Eigen::Vector6d state_vector =
//             utility::TransformMatrix4dToVector6d(transformation);
    const Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    // Accumulate JTJ/JTr by expanding each 3D residual into 3 scalar residuals.


    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                const size_t cs = corres[i][0];
                const size_t ct = corres[i][1];
                const Eigen::Vector3d &ps = source.points_[cs];
                const Eigen::Vector3d &pt = target.points_[ct];
                //source directions
                const Eigen::Vector3d &us = source_directions[cs];
                const Eigen::Vector3d &ut = target_directions[ct];
                const double &vs = source.dopplers_[cs];
                const double &vt = target.dopplers_[ct];

                J_r.resize(4);
                r.resize(4);
                w.resize(4);

                // // Compute predicted Doppler velocity.
                // const Eigen::Vector3d ds_in_S = R_S_to_V * ds_in_V;
                // const double doppler_pred_in_S = -ds_in_S.dot(v_s_in_S);
                // const double doppler_error = doppler_in_S - doppler_pred_in_S;

                // Dynamic point outlier pruning of correspondences.
                bool optimize{true};
                // if (reject_dynamic_outliers_ &&
                //     iteration >= outlier_rejection_min_iteration_ &&
                //     std::abs(doppler_error) > doppler_outlier_threshold_) {
                //     optimize = false;
                // }

                if (optimize) {
                    // Compute geometric point-to-point error and Jacobian.
                    const Eigen::Vector3d rg = ps - pt;


                    const Eigen::Matrix3d Jg_rot = -utility::SkewMatrix(ps);; // d(p')/dδθ with left-multiplicative update
                    // ---- doppler scalar residual: r_v = u_q^T * R * (s_p*u_p) - s_q
                    const Eigen::Vector3d v_p  = vs * us;     // source LOS velocity vector
                    const Eigen::Vector3d Rv_p = R * v_p;       // into target frame
                    const double rv = ut.dot(Rv_p) - vt;      // scalar residual
                    // Jacobian wrt rotation: d(u_q^T R v)/dδθ = u_q^T ([δθ]_x R v) = (Rv × u_q)^T δθ
                    const Eigen::Vector3d Jv_rot_vec = Rv_p.cross(ut);    // 3x1; as a 1x3 row later
                    // ---- pack 4 scalar residuals: 3 geometric + 1 doppler
                    // geometric x/y/z
                    for (int k = 0; k < 3; ++k) {
                       Eigen::Vector6d Jgk = Eigen::Vector6d::Zero();
                       Jgk.segment<3>(0) = sqrt_lambda_geometric * Jg_rot.row(k).transpose();   // rotation block
                       Jgk.segment<3>(3) = sqrt_lambda_geometric * Eigen::Vector3d::Unit(k);    // translation block
                       J_r[k] = Jgk;
                       r[k]   = rg(k);
                    }
                    // doppler scalar (scaled by sqrt(lambda))
                    Eigen::Vector6d Jvk = Eigen::Vector6d::Zero();
                    Jvk.segment<3>(0) = sqrt_lambda_doppler * Jv_rot_vec;  // 1x3 rotation row
                    // translation block remains zero
                    J_r[3] = Jvk;
                    r[3]   = sqrt_lambda_doppler * rv;
                } else {
                    r[0] = 0.F;
                    w[0] = 0.F;
                    J_r[0].block<3, 1>(0, 0) = Eigen::Vector3d::Zero();
                    J_r[0].block<3, 1>(3, 0) = Eigen::Vector3d::Zero();

                    r[1] = 0.F;
                    w[1] = 0.F;
                    J_r[1].block<3, 1>(0, 0) = Eigen::Vector3d::Zero();
                    J_r[1].block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
                }
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2;
    std::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

double TransformationEstimationForDopplerVelocityICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals()) return 0.0;
    double err = 0.0, r;
    for (const auto &c : corres) {
        r = (source.points_[c[0]] - target.points_[c[1]])
                    .dot(target.normals_[c[1]]);
        err += r * r;
    }
    return std::sqrt(err / (double)corres.size());
};

RegistrationResult RegistrationDopplerVelocityICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<Eigen::Vector3d> &source_directions,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForDopplerVelocityICP &estimation
        /* = TransformationEstimationForDopplerVelocityICP()*/,
        const ICPConvergenceCriteria &criteria /* = ICPConvergenceCriteria()*/,
        const double period /* = 0.1F*/,
        const Eigen::Matrix4d &T_V_to_S /* = Eigen::Matrix4d::Identity()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }

    if ((estimation.GetTransformationEstimationType() ==
         TransformationEstimationType::DopplerVelocityICP) &&
        (!target.HasNormals() || !source.HasDopplers())) {
        utility::LogError(
                "TransformationEstimationDopplerVelocityICP requires Doppler "
                "velocities for source PointCloud and pre-computed normal "
                "vectors for target PointCloud.");
    }

    Eigen::Matrix4d transformation = init;
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
    bool converged{false};
    for (i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);

        // Compute the transform update.
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_, source_directions,
                period, transformation, T_V_to_S, i);
        transformation = update * transformation;
        pcd.Transform(update);

        // Update the registration result.
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);

        // Check for convergence.
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
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
