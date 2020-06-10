/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#ifndef NONLINEAR_SQRT_TESTS_H
#define NONLINEAR_SQRT_TESTS_H


#include <functional>
#include "../tests.h"
#include "../independent-noise.h"
#include "../transformations.h"
//#include "transformations/LinearizedTransformation.h"

#include <gtest/gtest.h>

using namespace OpenKalman;

struct nonlinear_sqrt_tests : public ::testing::Test {
  nonlinear_sqrt_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~nonlinear_sqrt_tests() override {}
};


#endif //NONLINEAR_SQRT_TESTS_H
