/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#ifndef COVARIANCE_TESTS_H
#define COVARIANCE_TESTS_H

#include "../tests.h"
#include "distributions/GaussianDistribution.h"
#include "distributions/DistributionTraits.h"

struct covariance_tests : public ::testing::Test
{
  covariance_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~covariance_tests() override {}
};

#endif //COVARIANCE_TESTS_H
