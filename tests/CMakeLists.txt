# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

find_package(GTest)
if (GTest_FOUND)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})

    add_executable(covariances
        covariances/covariance_tests.cpp
        covariances/tests-covariance.cpp
        covariances/tests-covariance-ops.cpp
        covariances/tests-sqrt-covariance.cpp
        covariances/tests-sqrt-covariance-ops.cpp
        covariances/tests-covariance-refs.cpp
        )
    target_link_libraries(covariances ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(covariances 0)

    add_executable(typed-matrix
        typed-matrix/matrices.cpp
        typed-matrix/matrix.cpp
        typed-matrix/mean.cpp
        typed-matrix/euclidean_mean.cpp
        typed-matrix/matrix-refs.cpp
        typed-matrix/distributions.cpp
        )
    target_link_libraries(typed-matrix ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(typed-matrix 0)

    add_executable(transformation
        transformation/transformations.cpp
        transformation/general.cpp
        transformation/linear.cpp
        transformation/stochastic.cpp
        transformation/linear-fd.cpp
        )
    target_link_libraries(transformation ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(transformation 0)

    add_executable(transform-linear
        transform-linear/transform-linear.cpp
        transform-linear/linear-axis.cpp
        transform-linear/linear-angle.cpp
        transform-linear/linearized-add.cpp
        transform-linear/linearized-aug.cpp
        )
    target_link_libraries(transform-linear ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(transform-linear 0)

    add_executable(transform-nonlinear
        transform-nonlinear/transform-nonlinear.cpp
        transform-nonlinear/sample.cpp
        transform-nonlinear/sample-aug.cpp
        transform-nonlinear/rot-invariance.cpp
        transform-nonlinear/rev-rot-invariance.cpp
        )
    target_link_libraries(transform-nonlinear ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(transform-nonlinear 0)

    add_executable(linear
        linear/linear.cpp
        linear/linear.2x2.cpp
        linear/linear.2x3.cpp
        linear/sample.2x2.cpp
        linear/sample.2x3.cpp
        linear/sample.3x2.cpp
        linear/sample.4x3.cpp
        linear/identity.cpp
        linear/MCT.cpp
        )
    target_link_libraries(linear ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(linear 0)

    add_executable(nonlinear
        nonlinear/nonlinear.cpp
        nonlinear/TT.2x1.cpp
        nonlinear/TT.2x2.cpp
        nonlinear/CT.2x1.cpp
        nonlinear/CT.2x2.cpp
        nonlinear/UT.2x1.cpp
        nonlinear/UT.2x2.cpp
        nonlinear/MCT.2x2.cpp
        )
    target_link_libraries(nonlinear ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(nonlinear 0)

    add_executable(kalman
        kalman/kalman.cpp
        kalman/rotation.2D.cpp
        kalman/rotation.3D.cpp
        kalman/artillery.cpp
        kalman/radar.cpp
        )
    target_link_libraries(kalman ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(kalman 0)

ENDIF()