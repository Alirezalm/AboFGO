#pragma once

#include "Factor.h"

namespace AboFGO {

    class IMUFactor : public Factor {
    public:
        // measurement: 3D velocity measurement; weight: measurement weight.
        IMUFactor(const Eigen::Vector3d &measurement, double weight)
                : measurement_(measurement), weight_(weight) {}

        double error(const Eigen::VectorXd &state) const override {
            // state.segment<3>(3) is the velocity portion.
            Eigen::Vector3d r = state.segment<3>(3) - measurement_;
            return 0.5 * weight_ * r.squaredNorm();
        }

        void linearize(const Eigen::VectorXd &state, Eigen::MatrixXd &A, Eigen::VectorXd &b) const override {
            // Residual: r = velocity - measurement.
            Eigen::Vector3d r = state.segment<3>(3) - measurement_;
            // Jacobian with respect to state: [0_3, I_3].
            A = Eigen::MatrixXd::Zero(3, state.size());
            A.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();
            b = weight_ * r;
        }

    private:
        Eigen::Vector3d measurement_;
        double weight_;
    };

}