// src/main.cpp
#include "include/FactorGraph.h"
#include "include/GNSSFactor.h"
#include "include/IMUFactor.h"
#include "include/optimizer.h"
#include <iostream>
#include <Eigen/Dense>

int main() {
    // Create the factor graph.
    AboFGO::FactorGraph graph;

    // Simulated true state: position = [10, 20, 30], velocity = [1, 2, 3].
    Eigen::Vector3d gnss_measurement(10.0, 20.0, 30.0);
    Eigen::Vector3d imu_measurement(1.0, 2.0, 3.0);

    // Create factors with unit weight.
    auto gnssFactor = new AboFGO::GNSSFactor(gnss_measurement, 1.0);
    auto imuFactor = new AboFGO::IMUFactor(imu_measurement, 1.0);

    // Add factors to the graph.
    graph.addFactor(gnssFactor);
    graph.addFactor(imuFactor);

    // Print initial state and error.
    std::cout << "Initial state:\n" << graph.getState() << std::endl;
    std::cout << "Initial total error: " << graph.computeTotalError() << std::endl;

    // Create the optimizer and run optimization for 10 iterations.
    AboFGO::Optimizer optimizer(&graph);
    optimizer.optimize(10);

    // Print the final state.
    std::cout << "Final optimized state:\n" << graph.getState() << std::endl;

    return 0;
}
