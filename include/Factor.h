#pragma once

#include <Eigen/Dense>

namespace AboFGO {

    class Factor {
    public:
        virtual ~Factor() = default;

        // Compute the cost (0.5 * squared norm of weighted residual) given the current state.
        virtual double error(const Eigen::VectorXd &state) const = 0;

        // Compute the Jacobian (A) and weighted residual (b) for this factor at the given state.
        virtual void linearize(const Eigen::VectorXd &state, Eigen::MatrixXd &A, Eigen::VectorXd &b) const = 0;
    };

}
