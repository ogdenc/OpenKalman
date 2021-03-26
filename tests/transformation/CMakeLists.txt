# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

find_package(GTest)
if (GTest_FOUND)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})

    add_executable(transformation
        transformations.cpp
        general.cpp
        linear.cpp
        stochastic.cpp
        linear-fd.cpp
        )
    target_link_libraries(transformation ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(transformation 0)
ENDIF()