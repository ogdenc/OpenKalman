/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for coordinates::Polar
 */

#include "collections/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

using OpenKalman::stdcompat::numbers::pi;

TEST(coordinates, Polar)
{
  static_assert(descriptor<Polar<>>);
  static_assert(fixed_pattern<Polar<>>);
  static_assert(pattern<Polar<Distance, angle::Degrees>>);
  static_assert(not euclidean_pattern<Polar<>>);

  static_assert(get_dimension(Polar<>{}) == 2);
  static_assert(get_stat_dimension(Polar<>{}) == 3);
  static_assert(not get_is_euclidean(Polar<>{}));
  static_assert(dimension_of_v<Polar<>> == 2);
  static_assert(stat_dimension_of_v<Polar<>> == 3);

  static_assert(std::is_assignable_v<Polar<>&, Polar<>>);

  static_assert(std::is_same_v<std::common_type_t<Polar<Distance, angle::Degrees>,
    Polar<Distance, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>>>, Polar<Distance, angle::Degrees>>);
  static_assert(std::is_same_v<std::common_type_t<Polar<angle::PositiveDegrees, Distance>,
    Polar<Angle<std::integral_constant<int, 0>, std::integral_constant<int, 360>>, Distance>>, Polar<angle::PositiveDegrees, Distance>>);
}

#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/wrap.hpp"

namespace
{
  constexpr double sqrt3_2 = values::sqrt(3.)/2;
}

TEST(coordinates, Polar_transformations)
{
  static_assert(values::internal::near(to_stat_space(Polar{}, std::array{3., pi/3})[1U], 0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(Polar<angle::Degrees, Distance>{}, std::array{-420, -3})[0U], -0.5, 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(to_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{values::Fixed<double, 3>{}, values::Fixed<double, 60>{}}).get<1>())>, 0.5, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(to_stat_space(Polar<angle::PositiveDegrees, Distance>{}, std::tuple{values::Fixed<double, -420>{}, values::Fixed<double, -3>{}}).get<0>())>, -0.5, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(Polar{}, std::vector{3., pi/3}), std::array{3., 0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<angle::Radians, Distance>{}, std::array{pi/3, 3.}), std::array{0.5, sqrt3_2, 3.}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{3., 60}), std::array{3., 0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{3., 420}), std::array{3., 0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{-3., 420}), std::array{3., -0.5, -sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{3., -420}), std::array{3., 0.5, -sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{-3., -420}), std::array{3., -0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(Polar<Distance, angle::PositiveDegrees>{}, std::tuple{-3., -420}), std::array{3., -0.5, sqrt3_2}, 1e-6));

  static_assert(values::internal::near(from_stat_space(Polar{}, std::array{3., 0.5, sqrt3_2})[1U], pi/3, 1e-6));
  static_assert(values::internal::near(from_stat_space(Polar<angle::Degrees, Distance>{}, std::array{0.5, -sqrt3_2, 3.})[0U], -60, 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(from_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{values::Fixed<double, 3>{}, values::Fixed<double, 1>{}, values::Fixed<double, 1>{}}).get<1>())>, 45, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(from_stat_space(Polar<angle::PositiveDegrees, Distance>{}, std::tuple{values::Fixed<double, 1>{}, values::Fixed<double, -1>{}, values::Fixed<double, 3>{}}).get<0>())>, 315, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(Polar{}, std::tuple{3, 0.5, sqrt3_2}), std::array{3., pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Polar{}, std::tuple{3, 0.5, -sqrt3_2}), std::array{3., -pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Polar<Distance, angle::PositiveRadians>{}, std::tuple{3, 0.5, -sqrt3_2}), std::array{3., 5*pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{3, 0.5, sqrt3_2}), std::array{3, 60}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Polar<Distance, angle::Degrees>{}, std::tuple{3, 0.5, -sqrt3_2}), std::array{3, -60}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Polar<Distance, angle::PositiveDegrees>{}, std::tuple{3, 0.5, -sqrt3_2}), std::array{3, 300}, 1e-6));

  static_assert(values::internal::near(wrap(Polar{}, std::array{3., 1.1*pi})[1U], -0.9*pi, 1e-6));
  static_assert(values::internal::near(wrap(Polar<angle::Degrees, Distance>{}, std::array{190., 2.})[0U], -170., 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(wrap(Polar<Distance, angle::Degrees>{}, std::tuple{values::Fixed<double, -2>{}, values::Fixed<double, -190>{}}).get<1>())>, -10, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(wrap(Polar<angle::PositiveDegrees, Distance>{}, std::tuple{values::Fixed<double, -10>{}, values::Fixed<double, -2>{}}).get<0>())>, 170, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(Polar{}, std::vector{2., 1.1*pi}), std::array{2., -0.9*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar{}, std::tuple{-2, 1.1*pi}), std::array{2., 0.1*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar<angle::Degrees, Distance>{}, std::vector{190., 2.}), std::array{-170, 2}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar<angle::Degrees, Distance>{}, std::vector{-190., -2.}), std::array{-10, 2}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar<Distance, angle::PositiveRadians>{}, std::vector{-2., -0.1*pi}), std::array{2., 0.9*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar<Distance, angle::PositiveRadians>{}, std::vector{2., -0.1*pi}), std::array{2., 1.9*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar<angle::PositiveDegrees, Distance>{}, std::vector{-10, 2}), std::array{350, 2}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Polar<angle::PositiveDegrees, Distance>{}, std::vector{-10, -2}), std::array{170, 2}, 1e-6));
}
