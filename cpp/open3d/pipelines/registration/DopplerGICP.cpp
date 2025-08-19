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
// Altered from:
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2021 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/DopplerGICP.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Vector6d;
namespace {

// Small helper.
// inline Matrix3d Skew(const Vector3d &v) {
//     Matrix3d S;
//     S <<      0, -v.z(),  v.y(),
//            v.z(),     0, -v.x(),
//           -v.y(),  v.x(),     0;
//     return S;
// }
// inline Matrix3d InvSqrtPD(const Matrix3d &A,
//                           double clamp_lo = 1e-10,
//                           double clamp_hi = 1e+6) {
//     Eigen::SelfAdjointEigenSolver<Matrix3d> es(A);
//     const Matrix3d V = es.eigenvectors();
//     Eigen::Vector3d w = es.eigenvalues();
//     for (int k = 0; k < 3; ++k) {
//         w[k] = std::min(std::max(w[k], clamp_lo), clamp_hi);
//         w[k] = 1.0 / std::sqrt(w[k]);
//     }
//     return V * w.asDiagonal() * V.transpose();
// } 
/// Obtain the Rotation matrix that transform the basis vector e1 onto the
/// input vector x.
// inline Eigen::Matrix3d GetRotationFromE1ToX(const Eigen::Vector3d &x) {
//     const Eigen::Vector3d e1{1, 0, 0};
//     const Eigen::Vector3d v = e1.cross(x);
//     const double c = e1.dot(x);
//     if (c < -0.99) {
//         // Then means that x and e1 are in the same direction
//         return Eigen::Matrix3d::Identity();
//     }

//     const Eigen::Matrix3d sv = utility::SkewMatrix(v);
//     const double factor = 1 / (1 + c);
//     return Eigen::Matrix3d::Identity() + sv + (sv * sv) * factor;
// }

/// Compute the covariance matrix according to the original paper. If the input
/// has already pre-computed covariances returns immediately. If the input has
/// pre-computed normals but no covariances, compute the covariances from those
/// normals. If there is no covariances nor normals, compute each covariance
/// matrix following the original implementation of GICP using 20 NN.
// std::shared_ptr<geometry::PointCloud> InitializePointCloudForDopplerGICP(
//         const geometry::PointCloud &pcd, double epsilon) {
//     auto output = std::make_shared<geometry::PointCloud>(pcd);
//     if (output->HasCovariances()) {
//         utility::LogDebug("DopplerGICP: Using pre-computed covariances.");
//         return output;
//     }
//     else {
//         // Compute covariances the same way is done in the original GICP paper.
//         utility::LogDebug("DopplerGICP: Computing covariances from points.");
//         output->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(20));
//     }

//     output->covariances_.resize(output->points_.size());
//     const Eigen::Matrix3d C = Eigen::Vector3d(epsilon, 1, 1).asDiagonal();
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < (int)output->normals_.size(); i++) {
//         const auto Rx = GetRotationFromE1ToX(output->normals_[i]);
//         output->covariances_[i] = Rx * C * Rx.transpose();
//     }
//     return output;
// }
}  // namespace
double TransformationEstimationForDopplerGICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) {
        return 0.0;
    }
    double err = 0.0;
    for (const auto &c : corres) {
        const Eigen::Vector3d &vs = source.points_[c[0]];
        const Eigen::Matrix3d &Cs = source.covariances_[c[0]];
        const Eigen::Vector3d &vt = target.points_[c[1]];
        const Eigen::Matrix3d &Ct = target.covariances_[c[1]];
        const Eigen::Vector3d d = vs - vt;
        const Eigen::Matrix3d M = Ct + Cs;
        const Eigen::Matrix3d W = M.inverse().sqrt();
        err += d.transpose() * W * d;
    }
    return std::sqrt(err / (double)corres.size());
}
void TransformationEstimationForDopplerGICP::ComputeDGICPRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double &geometric_rmse_out,
        double &doppler_rmse_out) const {
    if (corres.empty()) {
        last_geometric_rmse_ = 0.0;
        last_doppler_rmse_ = 0.0;
        last_combined_rmse_ = 0.0;
        return;
    }
    double sum_g2 = 0.0;
    for (const auto &c : corres) {
        const Eigen::Vector3d &vs = source.points_[c[0]];
        const Eigen::Matrix3d &Cs = source.covariances_[c[0]];
        const Eigen::Vector3d &vt = target.points_[c[1]];
        const Eigen::Matrix3d &Ct = target.covariances_[c[1]];
        const Eigen::Vector3d d = vs - vt;
        const Eigen::Matrix3d M = Ct + Cs;
        const Eigen::Matrix3d W = M.inverse().sqrt();
        sum_g2 += d.transpose() * W * d;
    }
    const double N = static_cast<double>(corres.size());
    last_geometric_rmse_ = std::sqrt(sum_g2 / N);

      // --- Doppler RMSE (optional, requires context) ---
    double sum_v2_norm = 0.0;  // normalized by sigma_v (unitless)
    double sum_v2_mps  = 0.0;  // in (m/s)^2, for reporting
    bool have_doppler = source.HasNormals() && target.HasNormals() &&
                        source.HasDopplers() && target.HasDopplers();

    if (have_doppler) {
        for (const auto &c : corres) {
            const int is = c(0), it = c(1);
            const double sp = source.dopplers_[is];
            const double sq = target.dopplers_[it];
            const Eigen::Vector3d &up = source.normals_[is]; // LOS (already rotated with pcd)
            const Eigen::Vector3d &uq = target.normals_[it];
            const double rv = uq.dot(sp * up) - sq;  // m/s
            sum_v2_mps  += rv * rv;
            const double rvn = (sigma_v_ > 0 ? rv / sigma_v_ : rv); // unitless
            sum_v2_norm += rvn * rvn;
        }
        last_doppler_rmse_ = std::sqrt(sum_v2_mps / N);     // m/s
    } else {
        last_doppler_rmse_ = 0.0;
    }
    // Combined RMSE consistent with optimization objective:
    // L = sum ||W r_g||^2 + lambda * sum (rv / sigma_v)^2
    geometric_rmse_out = last_geometric_rmse_;
    doppler_rmse_out   = last_doppler_rmse_;
}


Eigen::Matrix4d
TransformationEstimationForDopplerGICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasCovariances() ||
        !source.HasCovariances()) {
        return Eigen::Matrix4d::Identity();
    }
    if (!source.HasDopplers() || !target.HasDopplers()) {
        utility::LogError("Doppler-GICP requires dopplers on both clouds");
    }
    if (!source.HasCovariances() || !target.HasCovariances()) {
        utility::LogError("Doppler-GICP requires covariances on both clouds");
    }
    if (!source.HasNormals() || !target.HasNormals()) {
        utility::LogError("Doppler-GICP requires normals on both clouds for directional Doppler");
    }
    const double sqrt_lambda = std::sqrt(lambda_doppler_);
    const double inv_sigma   = (sigma_v_ > 0 ? 1.0 / sigma_v_ : 1.0);
    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                const Eigen::Vector3d &vs = source.points_[corres[i][0]];
                const Eigen::Matrix3d &Cs = source.covariances_[corres[i][0]];
                const Eigen::Vector3d &vt = target.points_[corres[i][1]];
                const Eigen::Matrix3d &Ct = target.covariances_[corres[i][1]];
                const Eigen::Vector3d d = vs - vt;
                const Eigen::Matrix3d M = Ct + Cs;
                const Eigen::Matrix3d W = M.inverse().sqrt();
                // Geometric residual (3 rows): r_g_w = W (p - q)
                const Vector3d rg_w = W * d;

                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = -utility::SkewMatrix(vs);
                J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                J = W * J;

                // --- Doppler residuals ---
                const double ds = source.dopplers_[corres[i][0]];
                const double dt = target.dopplers_[corres[i][1]];
                const Eigen::Vector3d &up = source.normals_[corres[i][0]];
                const Eigen::Vector3d &uq = target.normals_[corres[i][1]];
                const Vector3d  Rvp = ds * up;
                const double rv  = uq.dot(Rvp) - dt;   // m/s
                const Vector3d  Jv_rot_vec = Rvp.cross(uq); // 3x1


                // Dynamic point outlier pruning of correspondences.
                bool optimize{true};
                if (reject_dynamic_outliers_ &&
                    std::abs(rv) > doppler_outlier_threshold_) {
                    optimize = false;
                }
                if (optimize) {
                    // Pack 4 scalar rows
                    constexpr int n_rows = 4;
                    J_r.resize(n_rows);
                    r.resize(n_rows);
                    w.resize(n_rows);
                    // Geometric rows 0..2
                    for (size_t i = 0; i < 3; ++i) {
                        r[i] = rg_w(i);
                        w[i] = geometric_kernel_ ? geometric_kernel_->Weight(r[i]) : 1.0;
                        J_r[i] = J.row(i);
                    }

                    // Doppler row 3 (normalized + weighted)
                    r[3] = sqrt_lambda * inv_sigma * rv; // unitless
                    w[3] = doppler_kernel_ ? doppler_kernel_->Weight(r[3]) : 1.0;
                    Eigen::Vector6d Jv = Eigen::Vector6d::Zero();
                    Jv.head<3>() = sqrt_lambda * inv_sigma * Jv_rot_vec; // rotation only
                    J_r[3] = Jv;
                } else {
                    // Fully zero-out all four rows when skipping this correspondence
                    J_r.resize(4);
                    r.resize(4);
                    w.resize(4);
                    for (int k = 0; k < 4; ++k) {
                        J_r[k].setZero();
                        r[k] = 0.0;
                        w[k] = 0.0;
                    }
                }
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2 = -1.0;
    std::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success = false;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

RegistrationResult RegistrationDopplerGICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForDopplerGICP
                &estimation /* = TransformationEstimationForDopplerGICP()*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {

    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }
    if (!source.HasDopplers() || !target.HasDopplers()) {
        utility::LogError(
                "Doppler-GICP requires doppler velocities on both source and "
                "target PointClouds.");
    }
    if (!source.HasCovariances() || !target.HasCovariances()) {
        utility::LogError(
                "Doppler-GICP requires covariances on both source and target PointClouds.");
    }
    if (!source.HasNormals() || !target.HasNormals()) {
        utility::LogError(
                "Doppler-GICP requires normals on both source and target PointClouds for directional Doppler.");
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
    double geo_rmse = 0.0, dop_rmse = 0.0;
    estimation.ComputeDGICPRMSE(pcd, target, result.correspondence_set_, geo_rmse, dop_rmse);
    result.inlier_rmse_ = geo_rmse + dop_rmse; // unitless (whitened)
    int i;
    bool converged{false};
    for (i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("Doppler-GICP iter #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);

        // Compute the transform update.
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target,result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);

        // Update the registration result.
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        // double geo_rmse = 0.0, dop_rmse = 0.0;
        estimation.ComputeDGICPRMSE(pcd, target, result.correspondence_set_, geo_rmse, dop_rmse);
        result.inlier_rmse_ = geo_rmse + dop_rmse; // unitless (whitened)

        // Check for convergence.
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            converged = true;
            break;
        }
    }
    // always print the result of geo and doppler
    utility::LogDebug(
            "Doppler-GICP converged: {:d} iterations, fitness {:.4f}, "
            "inlier_rmse {:.4f}, geometric_rmse {:.4f}, doppler_rmse {:.4f}",
            i, result.fitness_, result.inlier_rmse_, geo_rmse, dop_rmse);

    result.num_iterations_ = i;
    result.converged_ = converged;
    return result;


    // return RegistrationICP(
    //         *InitializePointCloudForDopplerGICP(source, estimation.epsilon_),
    //         *InitializePointCloudForDopplerGICP(target, estimation.epsilon_),
    //         max_correspondence_distance, init, estimation, criteria);
    }

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
