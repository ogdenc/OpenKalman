/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#ifndef TRANSFORM_LINEAR_TESTS_H
#define TRANSFORM_LINEAR_TESTS_H

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "../tests.h"

using namespace Eigen;

struct transform_tests : public ::testing::Test
{
  transform_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~transform_tests() override {}
};

#endif //TRANSFORM_LINEAR_TESTS_H
