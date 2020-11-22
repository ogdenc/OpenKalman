# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


find_package(GTest)
if (GTest_FOUND)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})

    add_executable(coefficients
            tests/coefficients.cpp
        tests/coefficient-functions.cpp
        )
    target_link_libraries(coefficients ${GTEST_BOTH_LIBRARIES} Threads::Threads)
    add_test(coefficients 0)
ENDIF()