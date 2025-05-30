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
 * \brief Tests for \ref dynamic_pattern objects
 */

#include "basics/tests/tests.hpp"
#include "basics/basics.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"
#include "linear-algebra/coordinates/functions/make_pattern_vector.hpp"
#include "linear-algebra/coordinates/views/comparison.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using numbers::pi;

TEST(coordinates, Any)
{
  static_assert(pattern<Any<double>>);
  static_assert(pattern<Any<double>>);
  static_assert(dynamic_pattern<Any<double>>);
  static_assert(dynamic_pattern<Any<float>>);
  static_assert(dynamic_pattern<Any<long double>>);
  static_assert(not fixed_pattern<Any<double>>);
  static_assert(not euclidean_pattern<Any<double>>);
  static_assert(descriptor<Any<double>>);
  static_assert(dimension_of_v<Any<double>> == dynamic_size);
  static_assert(stat_dimension_of_v<Any<float>> == dynamic_size);
  EXPECT_EQ(get_dimension(Any<double> {Dimensions{5}}), 5);
  EXPECT_EQ(get_dimension(Any<double> {Dimensions<3>{}}), 3);
  EXPECT_EQ(get_stat_dimension(Any<double> {std::integral_constant<int, 7>{}}), 7);
  EXPECT_TRUE(get_is_euclidean(Any<double> {Dimensions{5}}));
  EXPECT_EQ(get_dimension(Any<double> {Polar<>{}}), 2);
  EXPECT_EQ(get_stat_dimension(Any<double> {Polar<>{}}), 3);
  EXPECT_EQ(get_dimension(Any<double> {Spherical<>{}}), 3);
  EXPECT_EQ(get_stat_dimension(Any<double> {Spherical<>{}}), 4);
  EXPECT_FALSE(get_is_euclidean(Any<double> {Polar<>{}}));
}


TEST(coordinates, dynamic_pattern_traits)
{
  static_assert(pattern<std::vector<std::size_t>>);
  static_assert(dynamic_pattern<std::vector<std::size_t>>);
  static_assert(dynamic_pattern<comparison_view<std::vector<std::size_t>>>);
  static_assert(not fixed_pattern<std::vector<std::size_t>>);
  static_assert(euclidean_pattern<std::size_t>);
  static_assert(pattern<std::vector<Distance>>);
  static_assert(dynamic_pattern<std::vector<Distance>>);
  static_assert(not dynamic_pattern<std::array<Distance, 4>>);
  static_assert(not dynamic_pattern<std::tuple<Axis, Distance, Dimensions<3>>>);
  static_assert(dynamic_pattern<std::tuple<Axis, Distance, Dimensions<>>>);
  static_assert(dynamic_pattern<std::vector<Any<double>>>);
  static_assert(dynamic_pattern<comparison_view<std::vector<Any<double>>>>);
  static_assert(dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<>>> == dynamic_size);
  static_assert(dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<3>>> != dynamic_size);
  static_assert(stat_dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<>>> == dynamic_size);
  static_assert(stat_dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<3>>> != dynamic_size);
}


TEST(coordinates, dynamic_pattern_functions)
{
  EXPECT_EQ(get_dimension(std::tuple{4u}), 4u);
  EXPECT_EQ(get_dimension(std::tuple{Dimensions{0}}), 0u);
  EXPECT_EQ(get_dimension(std::tuple{Dimensions{5}}), 5u);
  EXPECT_EQ(get_dimension(std::vector{Axis{}}), 1u);
  EXPECT_EQ(get_dimension(std::vector{angle::Degrees{}}), 1u);
  EXPECT_EQ(get_stat_dimension(std::vector{angle::Degrees{}}), 2u);
  EXPECT_EQ(get_dimension(std::vector{Dimensions<5>{}}), 5u);
  EXPECT_EQ(get_dimension(std::vector{Dimensions{5}}), 5u);
  EXPECT_EQ(get_dimension(std::vector{Polar{}}), 2u);
  EXPECT_EQ(get_dimension(std::vector{Polar<Distance, angle::PositiveRadians>{}}), 2u);
  EXPECT_EQ(get_stat_dimension(std::vector{Polar<angle::PositiveDegrees, Distance>{}}), 3u);
  EXPECT_EQ(get_dimension(std::vector{Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3u);
  EXPECT_EQ(get_stat_dimension(std::vector{Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4u);

  EXPECT_EQ(get_dimension(std::tuple{Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7u);
  EXPECT_EQ(get_dimension(std::tuple{Dimensions{5}, std::integral_constant<unsigned, 1>{}, angle::Degrees{}}), 7u);
  EXPECT_EQ(get_stat_dimension(std::tuple{Axis{}, Dimensions{5}, angle::Degrees{}}), 8u);
  EXPECT_EQ(get_dimension(std::tuple{Axis{}, inclination::Radians{}, angle::Degrees{}, Dimensions{5}}), 8u);
  EXPECT_EQ(get_stat_dimension(std::tuple{Dimensions{5}, Axis{}, inclination::Radians{}, angle::Degrees{}}), 10u);

  EXPECT_EQ(get_stat_dimension(std::array{Any{Dimensions{5}}, Any{Axis{}}, Any{inclination::Radians{}}, Any{angle::Degrees{}}}), 10u);

  EXPECT_TRUE(get_is_euclidean(make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, Dimensions<5>{})));
  EXPECT_FALSE(get_is_euclidean(make_pattern_vector(angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<3>{})));
  EXPECT_FALSE(get_is_euclidean(make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5>{})));
}
