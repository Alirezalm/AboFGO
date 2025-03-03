cmake_minimum_required(VERSION 3.28)
project(AboFGO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

if (Eigen3_FOUND)
    message(STATUS "Eigen3 found")
    message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
endif ()

find_package(pybind11 REQUIRED)

if (pybind11_FOUND)
    message(STATUS "pybind11 found")
    message(STATUS "pybind11 include dir: ${pybind11_INCLUDE_DIR}")
endif ()

add_library(AboFGO SHARED
        src/GNSSFactor.cpp
        src/IMUFactor.cpp
        src/FactorGraph.cpp
        src/optimizer.cpp
)
target_link_libraries(AboFGO PUBLIC Eigen3::Eigen)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE AboFGO)

pybind11_add_module(pyabofgo python/pyabofgo.cpp)
target_link_libraries(pyabofgo PRIVATE AboFGO)
target_link_libraries(pyabofgo PRIVATE Eigen3::Eigen)

