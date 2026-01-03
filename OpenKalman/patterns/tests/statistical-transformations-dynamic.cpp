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


TEST(patterns, combo_axis_angle_dynamic)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Axis{}}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{angle::Radians{}}, std::array{pi/3}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{angle::Degrees{}}, std::array{30.}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Axis{}}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Axis{}, Axis{}}, std::array{3., 2.}), std::array{3., 2.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{angle::Degrees{}}, std::array{-30.}), std::array{sqrt3_2, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{angle::Radians{}}, Any{angle::PositiveRadians{}}}, std::array{pi/3, pi/6}), std::array{0.5, sqrt3_2, sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{angle::Radians{}}, Any{Dimensions<2>{}}, Any{angle::Radians{}}}, std::array{pi/3, 3., 4., pi/6}), std::array{0.5, sqrt3_2, 3., 4., sqrt3_2, 0.5}/*5u*/, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{Axis{}}, Any{angle::Radians{}}}, std::array{3., pi/3}), std::array{3., 0.5, sqrt3_2}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Axis{}}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Axis{}}, std::array{-3.}), std::array{-3.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{angle::Radians{}}, std::array{0.5, sqrt3_2}), std::array{pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{angle::Degrees{}}, std::array{0.5, sqrt3_2}), std::array{60.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{Axis{}}, Any{angle::Radians{}}}, std::array{3., 0.5, -sqrt3_2}), std::array{3., -pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{angle::Radians{}}, Any{Axis{}}}, std::array{0.5, sqrt3_2, 3.}), std::array{pi/3, 3.}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(std::vector{Axis{}}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Axis{}}, std::array{-3.}), std::array{-3.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{angle::Radians{}}, std::array{1.2*pi}), std::array{-0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{angle::PositiveRadians{}}, std::array{-0.2*pi}), std::array{1.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{angle::Degrees{}}, std::array{200.}), std::array{-160.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{angle::PositiveDegrees{}}, std::array{-20.}), std::array{340.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{angle::Circle{}}, std::array{-0.2}), std::array{0.8}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{Axis{}}, Any{angle::Radians{}}}, std::array{3., 1.2*pi}), std::array{3., -0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{angle::Radians{}}, Any{Axis{}}}, std::array{1.2*pi, 3.}), std::array{-0.8*pi, 3.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{angle::Radians{}}, Any{Axis{}}, Any{angle::Radians{}}, Any{Axis{}}}, std::array{1.2*pi, 5., 1.3*pi, 3.}), std::array{-0.8*pi, 5., -0.7*pi, 3.}, 1e-6));
}


TEST(patterns, combo_distance_inclination_dynamic)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Distance{}}, std::array{-3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{angle::Radians{}}, std::array{pi/3}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Distance{}, Distance{}}, std::array{3., -2.}), std::array{3., 2.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{inclination::Radians{}, inclination::Radians{}}, std::array{pi/3, pi/6}), std::array{0.5, sqrt3_2, sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{Distance{}}, Any{inclination::Radians{}}}, std::array{3., pi/3}), std::array{3., 0.5, sqrt3_2}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Distance{}}, std::array{3.}), std::array{3.}/*0u*/, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Distance{}}, std::array{-3.}), std::array{-3.}/*0u*/, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{inclination::Radians{}}, std::array{0.5, sqrt3_2}), std::array{pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{inclination::Degrees{}}, std::array{0.5, sqrt3_2}), std::array{60.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{Distance{}}, Any{inclination::Radians{}}}, std::array{3., 0.5, -sqrt3_2}), std::array{3., pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{inclination::Radians{}}, Any{Distance{}}}, std::array{0.5, sqrt3_2, 3.}), std::array{pi/3, 3.}, 1e-6));
  
  EXPECT_TRUE(test::is_near(wrap(std::vector{Distance{}}, std::array{3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Distance{}}, std::array{-3.}), std::array{3.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{inclination::Radians{}}, std::array{-0.7*pi}), std::array{0.7*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{inclination::Radians{}}, std::array{1.2*pi}), std::array{0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{inclination::Degrees{}}, std::array{-110.}), std::array{110.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{inclination::Degrees{}}, std::array{200.}), std::array{160.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{Distance{}}, Any{inclination::Radians{}}}, std::array{3., -0.7*pi}), std::array{3., 0.7*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{inclination::Radians{}}, Any{Distance{}}}, std::array{1.2*pi, 3.}), std::array{0.8*pi, 3.}, 1e-6));
}


TEST(patterns, combo_polar_dynamic)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Polar<Distance, angle::Radians>{}}, std::array{3., pi/3}), std::array{3., 0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Polar<angle::Radians, Distance>{}}, std::array{pi/3, 3.}), std::array{0.5, sqrt3_2, 3.}/*2u*/, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Polar<Distance, angle::Degrees>{}}, std::array{3., 60.}), std::array{3., 0.5, sqrt3_2}/*2u*/, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{Axis{}}, Any{Polar<Distance, angle::Degrees>{}}}, std::array{1.1, 3., 60.}), std::array{1.1, 3., 0.5, sqrt3_2}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Polar<Distance, angle::Radians>{}}, std::array{3., 0.5, sqrt3_2}), std::array{3., pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{Axis{}}, Any{Polar<Distance, angle::Radians>{}}}, std::array{1.1, 3., 0.5, -sqrt3_2}), std::array{1.1, 3., -pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{Polar<angle::Degrees, Distance>{}}, Any{Axis{}}}, std::array{0.5, sqrt3_2, 3., 1.1}), std::array{60., 3., 1.1}, 1e-6));
  
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{Axis{}}, Any{Polar<Distance, angle::Radians>{}}}, std::array{3., -2., 1.1*pi}), std::array{3., 2., 0.1*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{Polar<angle::Degrees, Distance>{}}, Any{Axis{}}}, std::array{190., -2., 4.}), std::array{10., 2., 4.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{Polar<Distance, angle::PositiveRadians>{}}, Any{Axis{}}}, std::array{-2., -0.1*pi, -4.}), std::array{2., 0.9*pi, -4.}/*2u*/, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(std::vector{Any{Axis{}}, Any{Polar<angle::PositiveDegrees, Distance>{}}}, std::array{5., -10., -2.}), std::array{5., 170., 2.}, 1e-6));
}


TEST(patterns, combo_spherical_dynamic)
{
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}, std::array{2., pi/6, pi/3}), std::array{2., 0.75, sqrt3_2/2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{Axis{}}, Any{Spherical<Distance, angle::Radians, inclination::Radians>{}}}, std::array{3., 2., pi/6, pi/3}), std::array{3., 2., 0.75, sqrt3_2/2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(std::vector{Any{Spherical<Distance, angle::Radians, inclination::Radians>{}}, Any{Axis{}}}, std::array{2., pi/6, pi/3, 3.}), std::array{2., 0.75, sqrt3_2/2, 0.5, 3.}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}, std::array{2., sqrt3_2/2, 0.25, sqrt3_2}), std::array{2., pi/6, pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{Axis{}}, Any{Spherical<Distance, angle::Radians, inclination::Radians>{}}}, std::array{3., 2., sqrt3_2/2, 0.25, sqrt3_2}), std::array{3., 2., pi/6, pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(std::vector{Any{Spherical<Distance, angle::Radians, inclination::Radians>{}}, Any{Axis{}}}, std::array{2., sqrt3_2/2, 0.25, sqrt3_2, 3.}), std::array{2., pi/6, pi/6, 3.}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}, std::array{-2., 1.1*pi, 0.6*pi}), std::array{2., 0.1*pi, 0.4*pi}, 1e-6));
}

