/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for statistical coordinate functions
 */

#include "collections/tests/tests.hpp"
#include "patterns/descriptors/Any.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Distance.hpp"
#include "patterns/descriptors/Angle.hpp"
#include "patterns/descriptors/Inclination.hpp"
#include "patterns/descriptors/Polar.hpp"
#include "patterns/descriptors/Spherical.hpp"
#include "patterns/functions/to_stat_space.hpp"
#include "patterns/functions/from_stat_space.hpp"
#include "patterns/functions/wrap.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;

using stdex::numbers::pi;

namespace
{
  constexpr double sqrt3_2 = values::sqrt(3.)/2;
}

TEST(patterns, combo_axis_angle_fixed)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Axis>{}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Axis, Axis>{}, std::array{3., 2.}), std::array{3., 2.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Dimensions<2>, Dimensions<3>>{}, std::array{3., 2., 5., 6., 7.}), std::array{3., 2., 5., 6., 7.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Axis, angle::Degrees>{}, std::array{4., -60.}), std::array{4., 0.5, -sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<angle::PositiveRadians, Axis>{}, std::array{pi/6, 4.}), std::array{sqrt3_2, 0.5, 4.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Dimensions<2>, angle::PositiveRadians, Axis>{}, std::array{1., 2., pi/6, 4.}), std::array{1., 2., sqrt3_2, 0.5, 4.}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<Axis, angle::Radians>{}, std::array{3., 0.5, -sqrt3_2}), std::array{3., -pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<angle::Radians, Axis>{}, std::array{0.5, -sqrt3_2, 4.}), std::array{-pi/3, 4.}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(std::tuple<Axis, angle::PositiveDegrees>{}, std::array{3., -60.}), std::array{3., 300.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::tuple<angle::Radians, Axis, angle::Radians, Axis>{}, std::array{1.2*pi, 5., 1.3*pi, 3.}), std::array{-0.8*pi, 5., -0.7*pi, 3.}, 1e-6));
}


TEST(patterns, combo_distance_inclination_fixed)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Distance>{}, std::array{-3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<inclination::Degrees, Distance>{}, std::array{-60., 2.}), std::array{0.5, sqrt3_2, 2.}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<Distance>{}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<Distance, inclination::Radians>{}, std::array{3., 0.5, -sqrt3_2}), std::array{3., pi/3}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(std::tuple<inclination::Degrees, Distance>{}, std::array{-60., -2.}), std::array{60., 2.}, 1e-6));
}


TEST(patterns, combo_polar_fixed)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Dimensions<2>, Polar<Distance, angle::Degrees>, Axis>{}, std::array{1.1, 2.1, 3., 60., 3.1}), std::array{1.1, 2.1, 3., 0.5, sqrt3_2, 3.1}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<Dimensions<2>, Polar<Distance, angle::Degrees>, Axis>{}, std::array{1.1, 2.1, 3., 0.5, sqrt3_2, 3.1}), std::array{1.1, 2.1, 3., 60., 3.1}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, std::array{3., -2., 1.1*pi}), std::array{3., 2., 0.1*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, std::array{190., 2., 4.}), std::array{-170., 2., 4.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::tuple<Polar<Distance, angle::PositiveRadians>, Axis>{}, std::array{-2., -0.1*pi, -4.}), std::array{2., 0.9*pi, -4.}, 1e-6));
}


TEST(patterns, combo_spherical_fixed)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, std::array{3., 2., pi/6, pi/3}), std::array{3., 2., .75, sqrt3_2/2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, std::array{2., pi/6, pi/3, 3.}), std::array{2., 0.75, sqrt3_2/2, 0.5, 3.}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, std::array{3., 2., sqrt3_2/2, 0.25, sqrt3_2}), std::array{3., 2., pi/6, pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, std::array{2., sqrt3_2/2, 0.25, sqrt3_2, 3.}), std::array{2., pi/6, pi/6, 3.}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, std::array{3., -2., 1.1*pi, 0.6*pi}), std::array{3., 2., 0.1*pi, 0.4*pi}, 1e-6));
}

