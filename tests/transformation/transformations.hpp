/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#ifndef TRANSFORMATION_TESTS_HPP
#define TRANSFORMATION_TESTS_HPP

#include <gtest/gtest.h>
#include "OpenKalman-Eigen3.hpp"

#include "basics/tests/tests.hpp"
#include "basics/tests/test-transformations.hpp"

struct transformations : public ::testing::Test
{
  transformations() {}

  void SetUp() override {}

  void TearDown() override {}

  ~transformations() override {}
};

#endif //TRANSFORMATION_TESTS_HPP
