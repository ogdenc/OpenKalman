# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED CONFIG)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(special-matrices
        ConstantAdapter.test.cpp
        constant-matrix-functions.test.cpp
        #DiagonalMatrix.test.cpp
        #SelfAdjointMatrix.test.cpp
        #TriangularMatrix.test.cpp
        #ToEuclideanExpr.test.cpp
        #FromEuclideanExpr.test.cpp
        rank-update.test.cpp
        blocks.test.cpp
        decompositions.test.cpp
    )

target_link_libraries(special-matrices ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(special-matrices DISCOVERY_TIMEOUT 20)
