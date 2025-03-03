#pragma once

#include "Factor.h"
#include <vector>
#include <Eigen/Dense>

namespace AboFGO {

    class FactorGraph {
    public:
        FactorGraph() {
            // Initialize the state as a 6D zero vector: [position (3); velocity (3)].
            state_ = Eigen::VectorXd::Zero(6);
        }

        ~FactorGraph() {
            // Clean up all allocated factors.
            for (auto factor: factors_) {
                delete factor;
            }
        }

        // Add a factor to the graph.
        void addFactor(Factor *factor) {
            factors_.push_back(factor);
        }

        // Compute the total cost over all factors given the current state.
        double computeTotalError() const {
            double totalError = 0.0;
            for (const auto &factor: factors_) {
                totalError += factor->error(state_);
            }
            return totalError;
        }

        // Accessor to the state.
        Eigen::VectorXd &getState() { return state_; }

        const Eigen::VectorXd &getState() const { return state_; }

        // Accessor to the factors.
        const std::vector<Factor *> &getFactors() const { return factors_; }

    private:
        Eigen::VectorXd state_;
        std::vector<Factor *> factors_;
    };

}