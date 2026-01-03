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
 * \brief Tests for patterns::Angle
 */

#include "collections/tests/tests.hpp"
#include "patterns/concepts/fixed_pattern.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/functions/get_stat_dimension.hpp"
#include "patterns/functions/get_is_euclidean.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/traits/stat_dimension_of.hpp"
#include "patterns/descriptors/Angle.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;

using OpenKalman::stdex::numbers::pi;

TEST(patterns, Angle)
{
  static_assert(descriptor<angle::Radians>);
  static_assert(fixed_pattern<angle::Radians>);
  static_assert(pattern<angle::Degrees>);
  static_assert(not euclidean_pattern<angle::Radians>);

  static_assert(get_dimension(angle::Radians{}) == 1);
  static_assert(get_stat_dimension(angle::Radians{}) == 2);
  static_assert(not get_is_euclidean(angle::Radians{}));
  static_assert(dimension_of_v<angle::Radians> == 1);
  static_assert(stat_dimension_of_v<angle::Radians> == 2);

  using ARef = stdex::reference_wrapper<angle::Radians>;
  static_assert(dimension_of_v<ARef> == 1);
  static_assert(stat_dimension_of_v<ARef> == 2);
  static_assert(not euclidean_pattern<ARef>);

  static_assert(std::is_same_v<std::common_type_t<angle::Degrees, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>>, angle::Degrees>);
}

#include "patterns/functions/to_stat_space.hpp"
#include "patterns/functions/from_stat_space.hpp"
#include "patterns/functions/wrap.hpp"

namespace
{
  constexpr double sqrt3_2 = values::sqrt(3.)/2;
}

TEST(patterns, Angle_transformations)
{
  static_assert(values::internal::near(to_stat_space(angle::Radians{}, std::array{pi/3})[0U], 0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(angle::Degrees{}, std::array{60})[0U], 0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(angle::PositiveRadians{}, std::array{4*pi/3})[0U], -0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(angle::PositiveDegrees{}, std::array{240})[0U], -0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(angle::Circle{}, std::array{-13./12})[1U], -0.5, 1e-6));

  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<0>(to_stat_space(angle::Degrees{}, std::array{values::fixed_value<double, 60>{}})))>, 0.5, 1e-6));
  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<0>(to_stat_space(angle::PositiveDegrees{}, std::tuple{values::fixed_value<double, 240>{}})))>, -0.5, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(angle::Radians{}, std::vector{pi/3}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::PositiveRadians{}, std::array{pi/3}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Degrees{}, std::tuple{60}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Degrees{}, std::vector{30}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Degrees{}, std::vector{390}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Degrees{}, std::array{-210}), std::array{-sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Degrees{}, std::array{-330}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Degrees{}, std::array{-390}), std::array{sqrt3_2, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::PositiveDegrees{}, std::array{390}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::PositiveDegrees{}, std::array{-30}), std::array{sqrt3_2, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::PositiveDegrees{}, std::array{-390}), std::array{sqrt3_2, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Circle{}, std::array{13./12}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Circle{}, std::array{17./12}), std::array{-sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(angle::Circle{}, std::array{-13./12}), std::array{sqrt3_2, -0.5}, 1e-6));

  static_assert(values::internal::near(from_stat_space(angle::Radians{}, std::array{0.5, sqrt3_2})[0U], pi/3, 1e-6));
  static_assert(values::internal::near(from_stat_space(angle::Degrees{}, std::array{0.5, -sqrt3_2})[0U], -60, 1e-6));
  static_assert(values::internal::near(from_stat_space(angle::PositiveRadians{}, std::array{-0.5, sqrt3_2})[0U], 2*pi/3, 1e-6));
  static_assert(values::internal::near(from_stat_space(angle::PositiveDegrees{}, std::array{-0.5, -sqrt3_2})[0U], 240, 1e-6));
  static_assert(values::internal::near(from_stat_space(angle::Circle{}, std::array{0.5, -sqrt3_2})[0U], 5./6, 1e-6));

  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<0>(from_stat_space(angle::Degrees{}, std::tuple{values::fixed_value<double, 1>{}, values::fixed_value<double, 1>{}})))>, 45, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(angle::Radians{}, std::vector{0.5, sqrt3_2}), std::array{pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(angle::Degrees{}, std::array{0.5, -sqrt3_2}), std::array{-60}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(angle::PositiveDegrees{}, std::tuple{0.5, -sqrt3_2}), std::array{300}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(angle::Degrees{}, std::array{-sqrt3_2, -0.5}), std::array{-150}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(angle::PositiveDegrees{}, std::array{-sqrt3_2, -0.5}), std::array{210}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(angle::Circle{}, std::array{-sqrt3_2, -0.5}), std::array{7./12}, 1e-6));

  static_assert(values::internal::near(wrap(angle::Radians{}, std::array{3.2*pi})[0U], -0.8*pi, 1e-6));
  static_assert(values::internal::near(wrap(angle::Degrees{}, std::array{200})[0U], -160, 1e-6));
  static_assert(values::internal::near(wrap(angle::PositiveRadians{}, std::array{-0.2*pi})[0U], 1.8*pi, 1e-6));
  static_assert(values::internal::near(wrap(angle::PositiveDegrees{}, std::array{-20})[0U], 340, 1e-6));
  static_assert(values::internal::near(wrap(angle::Circle{}, std::array{-0.2})[0U], 0.8, 1e-6));

  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<0>(wrap(angle::Degrees{}, std::array{values::fixed_value<double, 200>{}})))>, -160, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(angle::Radians{}, std::vector{1.2*pi}), std::array{-0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(angle::Radians{}, std::vector{3.2*pi}), std::array{-0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(angle::PositiveRadians{}, std::vector{-0.2*pi}), std::array{1.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(angle::PositiveRadians{}, std::vector{-2.2*pi}), std::array{1.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(angle::Degrees{}, std::vector{200}), std::array{-160}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(angle::PositiveDegrees{}, std::vector{-20}), std::array{340}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(angle::Circle{}, std::vector{-0.2}), std::array{0.8}, 1e-6));

  static constexpr auto c = angle::Circle{};
  EXPECT_TRUE(values::internal::near(to_stat_space(stdex::cref(c), std::array{-13./12})[1U], -0.5, 1e-6));
  EXPECT_TRUE(values::internal::near(from_stat_space(stdex::cref(c), std::array{0.5, -sqrt3_2})[0U], 5./6, 1e-6));
  EXPECT_TRUE(values::internal::near(wrap(stdex::cref(c), std::array{-0.2})[0U], 0.8, 1e-6));
}

