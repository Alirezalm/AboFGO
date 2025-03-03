#pragma once

#include "FactorGraph.h"
#include <Eigen/Dense>
#include <iostream>

namespace AboFGO {

    class Optimizer {
    public:
        Optimizer(FactorGraph *graph) : graph_(graph) {}

        void optimize(int iterations = 10) {
            Eigen::VectorXd &state = graph_->getState();
            int state_dim = state.size();

            for (int iter = 0; iter < iterations; ++iter) {
                // First, determine the total number of residuals.
                int total_rows = 0;
                for (const auto &factor: graph_->getFactors()) {
                    Eigen::MatrixXd A_i;
                    Eigen::VectorXd b_i;
                    factor->linearize(state, A_i, b_i);
                    total_rows += A_i.rows();
                }

                // Build the stacked Jacobian and residual vector.
                Eigen::MatrixXd A(total_rows, state_dim);
                Eigen::VectorXd b(total_rows);

                int current_row = 0;
                for (const auto &factor: graph_->getFactors()) {
                    Eigen::MatrixXd A_i;
                    Eigen::VectorXd b_i;
                    factor->linearize(state, A_i, b_i);
                    int rows = A_i.rows();
                    A.block(current_row, 0, rows, state_dim) = A_i;
                    b.segment(current_row, rows) = b_i;
                    current_row += rows;
                }

                // Compute the Gauss–Newton update:
                // Solve (AᵀA) delta = -Aᵀb.
                Eigen::MatrixXd H = A.transpose() * A;
                Eigen::VectorXd g = A.transpose() * b;
                Eigen::VectorXd delta = -H.ldlt().solve(g);

                // Update the state.
                state += delta;

                double err = graph_->computeTotalError();
                std::cout << "Iteration " << iter << ": total error = " << err << std::endl;
            }
        }

    private:
        FactorGraph *graph_;
    };

}