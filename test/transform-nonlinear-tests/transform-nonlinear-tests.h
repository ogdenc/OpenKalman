/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef TRANSFORM_NONLINEAR_TESTS_H
#define TRANSFORM_NONLINEAR_TESTS_H

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "../tests.h"
#include "../transformations.h"

struct transform_nonlinear_tests : public ::testing::Test
{
  transform_nonlinear_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~transform_nonlinear_tests() override {}

  template<
    typename Transform,
    typename Transformation,
    typename ReverseTransformation,
    typename AngleDist,
    typename AngleRotDist,
    typename ... Noise>
  ::testing::AssertionResult rotational_invariance_test(
    const Transform& t,
    const Transformation& f1,
    const ReverseTransformation& f2,
    const AngleDist& angle_input,
    const AngleRotDist& angle_input_rot,
    const Noise& ... noise)
  {
    const auto[output, cross] = t.transform_with_cross_covariance(angle_input, f1, noise...);
    const auto[output_rot, cross_rot] = t.transform_with_cross_covariance(angle_input_rot, f1, noise...);
    const auto res1 = is_near(
      f2(mean(output)),
      Mean {f2(mean(output_rot)) + make_Mean<typename DistributionTraits<AngleRotDist>::Mean::Coefficients>(0, M_PI)},
      1e-4);
    const auto res2 = is_near(covariance(output), covariance(output_rot), 1e-3);
    const auto res3 = is_near(cross, -cross_rot, 1e-4);
    if (res1 and res2 and res3) return ::testing::AssertionSuccess();
    else
      return ::testing::AssertionFailure() <<
        (res1 ? "" : "mean") << (res1 ? "" : res1.message()) <<
        (res2 ? "" : "covariance") << (res2 ? "" : res2.message()) <<
        (res3 ? "" : "cross covariance") << (res3 ? "" : res3.message());
  }


  template<
    typename Transform,
    typename Transformation,
    typename ReverseTransformation,
    typename LocDist,
    typename LocRotDist,
    typename ... Noise>
  ::testing::AssertionResult reverse_rotational_invariance_test(
    const Transform& t,
    const Transformation& f1,
    const ReverseTransformation& f2,
    const LocDist& loc_input,
    const LocRotDist& loc_input_rot,
    const Noise& ... noise)
  {
    const auto[output, cross] = t.transform_with_cross_covariance(loc_input, f1, noise...);
    const auto[output_rot, cross_rot] = t.transform_with_cross_covariance(loc_input_rot, f1, noise...);
    const auto res1 = is_near(
      f2(mean(output) - make_Mean<Polar<>>(0, M_PI)),
      Mean {f2(mean(output_rot))},
      1e-3);
    const auto res2 = is_near(covariance(output), covariance(output_rot), 1e-3);
    const auto res3 = is_near(cross, -cross_rot, 1e-3);
    if (res1 and res2 and res3) return ::testing::AssertionSuccess();
    else
      return ::testing::AssertionFailure() <<
        (res1 ? "" : "mean") << (res1 ? "" : res1.message()) <<
        (res2 ? "" : "covariance") << (res2 ? "" : res2.message()) <<
        (res3 ? "" : "cross covariance") << (res3 ? "" : res3.message());
  }

};


#endif //TRANSFORM_NONLINEAR_TESTS_H
