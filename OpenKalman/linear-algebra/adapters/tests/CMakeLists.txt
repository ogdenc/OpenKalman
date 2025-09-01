# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(adapters
        constant_adapter.cpp
        #constant-matrix-functions.test.cpp
        #diagonal_adapter.test.cpp
        #HermitianAdapter.test.cpp
        #TriangularAdapter.test.cpp
        #VectorSpaceAdapter.test.cpp
        #ToEuclideanExpr.test.cpp
        #FromEuclideanExpr.test.cpp
        #rank-update.test.cpp
        #blocks.test.cpp
        #linear-algebra.test.cpp
        #decompositions.test.cpp
    )

target_link_libraries(adapters GTest::gtest_main Eigen3::Eigen Threads::Threads)

gtest_discover_tests(adapters DISCOVERY_TIMEOUT 20)
