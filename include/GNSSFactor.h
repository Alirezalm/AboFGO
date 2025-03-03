
#pragma once

#include "Factor.h"

namespace AboFGO {

    class GNSSFactor : public Factor {
    public:
        // measurement: 3D GNSS position; weight: measurement weight.
        GNSSFactor(Eigen::Vector3d measurement, double weight)
                : measurement_(std::move(measurement)), weight_(weight) {}

        double error(const Eigen::VectorXd &state) const override {
            // state.head<3>() is the position portion.
            Eigen::Vector3d r = state.head<3>() - measurement_;
            return 0.5 * weight_ * r.squaredNorm();
        }

        void linearize(const Eigen::VectorXd &state, Eigen::MatrixXd &A, Eigen::VectorXd &b) const override {
            // Residual: r = position - measurement.
            Eigen::Vector3d r = state.head<3>() - measurement_;
            // Jacobian with respect to state: [I_3, 0_3x3]
            A = Eigen::MatrixXd::Zero(3, state.size());
            A.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            // Weight the residual.
            b = weight_ * r;
        }

    private:
        Eigen::Vector3d measurement_;
        double weight_;
    };

}