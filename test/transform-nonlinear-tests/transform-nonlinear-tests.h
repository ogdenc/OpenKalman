/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef TRANSFORM_TESTS_H
#define TRANSFORM_TESTS_H

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "../tests.h"
#include "transforms/OpenKalman-transforms.h"

struct transform_tests : public ::testing::Test
{
  transform_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~transform_tests() override {}

  template<
    typename SamplePointsType,
    typename Transformation,
    typename ReverseTransformation,
    typename AngleDist,
    typename AngleRotDist,
    typename ... Noise>
  ::testing::AssertionResult rotational_invariance_test(
    const Transformation& f1,
    const ReverseTransformation& f2,
    const AngleDist& angle_input,
    const AngleRotDist& angle_input_rot,
    const Noise& ... noise)
  {
    const auto t = make_SamplePointsTransform<SamplePointsType>(f1);
    const auto[output, cross] = t(angle_input, noise...);
    const auto[output_rot, cross_rot] = t(angle_input_rot, noise...);
    const auto res1 = is_near(f2(mean(output)), f2(mean(output_rot)) +
      make_Mean<typename DistributionTraits<AngleRotDist>::Mean::Coefficients>(0, M_PI));
    const auto res2 = is_near(covariance(output), covariance(output_rot));
    const auto res3 = is_near(cross, -cross_rot);
    if (res1 and res2 and res3) return ::testing::AssertionSuccess();
    else
      return ::testing::AssertionFailure() <<
        (res1 ? "" : "mean") << (res1 ? "" : res1.message()) <<
        (res2 ? "" : "covariance") << (res2 ? "" : res2.message()) <<
        (res3 ? "" : "cross covariance") << (res3 ? "" : res3.message());
  }

};

using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;
using P2 = Coefficients<Polar<>>;

static auto polar2Cartesian = make_Transformation<P2, C2>(
  [](const Mean<P2>& x, const auto&...noise) -> Mean<C2>
    {
      return {((x(0)*cos(x(1))) + ... + noise(0)), ((x(0)*sin(x(1))) + ... + noise(1))};
    });

static auto Cartesian2polar = make_Transformation<C2, P2>(
  [](const Mean<C2>& a, const auto&...noise) -> Mean<P2>
    {
      return {((std::hypot(a(1), a(0))) + ... + noise(0)), ((std::atan2(a(1), a(0))) + ... + noise(1))};
    });

static auto polar2polar = make_Transformation<P2, P2>(
  [](const Mean<P2> a, const auto&...noise) -> Mean<P2>
    {
      return ((a + Mean<P2>(0, M_PI)) + ... + noise);
    });

const GaussianDistribution angle_input {Mean<P2>(1, 0.95 * M_PI), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 9)};
const GaussianDistribution angle_input_rot {Mean<P2>(1, 0.95 * M_PI - M_PI), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 9)};
const GaussianDistribution angle_noise {Mean<P2>::zero(), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 81)};
const GaussianDistribution cart_noise {Mean<C2>::zero(), Covariance<C2>(0.01, 0, 0, 0.01)};


#endif //TRANSFORM_TESTS_H
