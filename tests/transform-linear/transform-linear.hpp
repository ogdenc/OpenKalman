/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#ifndef TRANSFORM_LINEAR_TESTS_HPP
#define TRANSFORM_LINEAR_TESTS_HPP

#include <gtest/gtest.h>
#include "OpenKalman-Eigen3.hpp"

#include "basics/tests/tests.hpp"

using namespace OpenKalman;

struct transform_linear : public ::testing::Test
{
  transform_linear() {}

  void SetUp() override {}

  void TearDown() override {}

  ~transform_linear() override {}
};

#endif //TRANSFORM_LINEAR_TESTS_HPP
