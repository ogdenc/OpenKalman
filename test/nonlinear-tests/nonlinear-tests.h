/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NONLINEAR_TESTS_H
#define NONLINEAR_TESTS_H

#include <functional>
#include <gtest/gtest.h>
#include "../tests.h"
#include "../transformations.h"

using namespace OpenKalman;

class nonlinear_tests : public ::testing::Test {
public:
    nonlinear_tests() {
    }

    void SetUp() override {
        // code here will execute just before the test ensues
    }

    void TearDown() override {
        // code here will be called just after the test completes
        // ok to throw exceptions from here if need be
    }

    ~nonlinear_tests() override {
        // cleanup any pending stuff, but no exceptions allowed
    }

    // put in any custom members that you need

};


#endif //NONLINEAR_TESTS_H
