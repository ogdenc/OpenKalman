# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.7)

#set(BOOST_USE_STATIC_LIBS        ON) # only find static libs
#set(BOOST_USE_MULTITHREADED      ON)

enable_testing()
find_package(GTest REQUIRED)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(boost 1.36.0 COMPONENTS optional REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
#include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
#include_directories(SYSTEM ${OPENCV_INCLUDE_DIRS})
#include_directories(${BOOST_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -g -O2 -ftemplate-backtrace-limit=0")

add_executable(matrix-tests
        matrix-tests/matrix_tests.cpp
        matrix-tests/tests-coefficients.cpp
        matrix-tests/tests-matrix.cpp
        matrix-tests/tests-diagonal.cpp
        matrix-tests/tests-self-adjoint.cpp
        matrix-tests/tests-triangular.cpp
        matrix-tests/tests-ToEuclideanExpr.cpp
        matrix-tests/tests-FromEuclideanExpr.cpp
        )
set_target_properties(matrix-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(matrix-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(matrix-tests 0)


add_executable(covariance-tests
        covariance-tests/covariance_tests.cpp
        covariance-tests/tests-covariance.cpp
        covariance-tests/tests-covariance-ops.cpp
        covariance-tests/tests-sqrt-covariance.cpp
        covariance-tests/tests-sqrt-covariance-ops.cpp
        covariance-tests/tests-covariance-refs.cpp
        )
set_target_properties(covariance-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(covariance-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(covariance-tests 0)


add_executable(typed-matrix-tests
        typed-matrix-tests/typed_matrix_tests.cpp
        typed-matrix-tests/tests-typed-matrix.cpp
        typed-matrix-tests/tests-mean.cpp
        typed-matrix-tests/tests-euclidean_mean.cpp
        typed-matrix-tests/tests-typed-matrix-refs.cpp
        typed-matrix-tests/tests-distributions.cpp
        )
set_target_properties(typed-matrix-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(typed-matrix-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(typed-matrix-tests 0)


add_executable(transformation-tests
        transformation-tests/transformation-tests.main.cpp
        transformation-tests/tests-general.cpp
        transformation-tests/tests-linear.cpp
        transformation-tests/tests-stochastic.cpp
        )
set_target_properties(transformation-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(transformation-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(transformation-tests 0)


add_executable(transform-linear-tests
        transform-linear-tests/transform-linear-tests.cpp
        transform-linear-tests/tests-linear-axis.cpp
        transform-linear-tests/tests-linear-angle.cpp
        transform-linear-tests/tests-linearized-add.cpp
        transform-linear-tests/tests-linearized-aug.cpp
        )
set_target_properties(transform-linear-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(transform-linear-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(transform-linear-tests 0)


add_executable(transform-nonlinear-tests
        transform-nonlinear-tests/transform-nonlinear-tests.cpp
        transform-nonlinear-tests/tests-sample.cpp
        transform-nonlinear-tests/tests-sample-aug.cpp
        transform-nonlinear-tests/tests-sample-angle.cpp
        )
set_target_properties(transform-nonlinear-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(transform-nonlinear-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(transform-nonlinear-tests 0)


add_executable(linear-tests
        linear-tests/linear-tests.main.cpp
        linear-tests/linear-tests.2x2.cpp
        linear-tests/linear-tests.2x2.sample.cpp
        linear-tests/linear-tests.3x2.sample.cpp
        linear-tests/linear-tests.4x3.sample.cpp
        )
set_target_properties(linear-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(linear-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(linear-tests 0)


add_executable(nonlinear-tests
        nonlinear-tests/nonlinear-tests.main.cpp
        nonlinear-tests/nonlinear-tests.2x1.CT.cpp
        nonlinear-tests/nonlinear-tests.2x1.TT.cpp
        nonlinear-tests/nonlinear-tests.2x1.UT.cpp
        nonlinear-tests/nonlinear-tests.2x2.CT.cpp
        nonlinear-tests/nonlinear-tests.2x2.MCT.cpp
        nonlinear-tests/nonlinear-tests.2x2.TT.cpp
        nonlinear-tests/nonlinear-tests.2x2.UT.cpp
        )
set_target_properties(nonlinear-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(nonlinear-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(nonlinear-tests 0)


add_executable(nonlinear-sqrt-tests
        nonlinear-sqrt-tests/nonlinear-sqrt-tests.main.cpp
        nonlinear-sqrt-tests/nonlinear-sqrt-tests.TT.cpp
        nonlinear-sqrt-tests/nonlinear-sqrt-tests.UT.symm.cpp
        nonlinear-sqrt-tests/nonlinear-sqrt-tests.UT.spher.cpp
        nonlinear-sqrt-tests/nonlinear-sqrt-tests.CT.cpp
        )
set_target_properties(nonlinear-sqrt-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(nonlinear-sqrt-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(nonlinear-sqrt-tests 0)


add_executable(MonteCarlo-tests
        MonteCarlo-tests/MonteCarlo-tests.2x1.cpp
        MonteCarlo-tests/MonteCarlo-tests.2x2.cpp
        )
set_target_properties(MonteCarlo-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(MonteCarlo-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(MonteCarlo-tests 0)


add_executable(kalman-tests
        kalman-tests/kalman-tests.main.cpp
        kalman-tests/kalman-tests.2x2.update.cpp
        kalman-tests/kalman-tests.2x2.linear.cpp
        kalman-tests/kalman-tests.2x2.CT.Add.cpp
        kalman-tests/kalman-tests.2x2.CT.Aug.cpp
        kalman-tests/kalman-tests.2x2.UT.spher.Add.cpp
        kalman-tests/kalman-tests.radial.CT.Aug.cpp
        )
set_target_properties(kalman-tests PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
target_link_libraries(kalman-tests ${GTEST_BOTH_LIBRARIES} Threads::Threads)
add_test(kalman-tests 0)
