# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


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
