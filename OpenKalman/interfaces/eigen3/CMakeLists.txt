# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

find_package(GTest)
if (GTest_FOUND)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})

    add_executable(eigen3
        tests/eigen3.gtest.cpp
        tests/eigen3-native-matrix.test.cpp
        tests/DiagonalMatrix.test.cpp
        tests/SelfAdjointMatrix.test.cpp
        tests/TriangularMatrix.test.cpp
        tests/ToEuclideanExpr.test.cpp
        tests/FromEuclideanExpr.test.cpp
        )
    target_link_libraries(eigen3 ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(eigen3 0)
ENDIF()
