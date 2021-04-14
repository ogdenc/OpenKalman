/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NONLINEAR_TESTS_HPP
#define NONLINEAR_TESTS_HPP

#include "../tests.hpp"
#include "../test-transformations.hpp"

using namespace OpenKalman;

class nonlinear : public ::testing::Test {
public:
    nonlinear() {
    }

    void SetUp() override {
        // code here will execute just before the test ensues
    }

    void TearDown() override {
        // code here will be called just after the test completes
        // ok to throw exceptions from here if need be
    }

    ~nonlinear() override {
        // cleanup any pending stuff, but no exceptions allowed
    }

    // put in any custom members that you need

};


#endif //NONLINEAR_TESTS_HPP
