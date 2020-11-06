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

    add_executable(typed-matrix
        matrices.cpp
        matrix.cpp
        mean.cpp
        euclidean_mean.cpp
        matrix-refs.cpp
        distributions.cpp
        )
    target_link_libraries(typed-matrix ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(typed-matrix 0)
ENDIF()