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

#include "collections/tests/tests.hpp"
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

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using stdcompat::numbers::pi;

namespace
{
  constexpr double sqrt3_2 = values::sqrt(3.)/2;
  constexpr double sqrt3_4 = values::sqrt(3.)/4;
}

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

  EXPECT_NEAR((to_stat_space(Any{Dimensions{5}}, std::array{1., 2., 3., 4., 5.})[1]), 2., 1e-6);
  EXPECT_NEAR((to_stat_space(Any{Dimensions<5>{}}, std::array{1., 2., 3., 4., 5.})[2]), 3., 1e-6);
  EXPECT_NEAR((from_stat_space(Any{Dimensions<5>{}}, std::array{1., 2., 3., 4., 5.})[3]), 4., 1e-6);
  EXPECT_NEAR((wrap(Any{Dimensions<5>{}}, std::array{1., 2., 3., 4., 5.})[4]), 5., 1e-6);

  EXPECT_NEAR(to_stat_space(Any{Distance{}}, std::array{-3.})[0U], 3., 1e-6);
  EXPECT_NEAR(from_stat_space(Any{Distance{}}, std::array{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(wrap(Any{Distance{}}, std::array{-3.})[0U], 3., 1e-6);

  EXPECT_NE(OpenKalman::coordinates::internal::get_descriptor_hash_code(angle::Radians{}), OpenKalman::coordinates::internal::get_descriptor_hash_code(angle::Degrees{}));
  EXPECT_NE(OpenKalman::coordinates::internal::get_descriptor_hash_code(Any{angle::Radians{}}), OpenKalman::coordinates::internal::get_descriptor_hash_code(Any{angle::Degrees{}}));
  EXPECT_NE(OpenKalman::coordinates::internal::get_descriptor_hash_code(Any{inclination::Radians{}}), OpenKalman::coordinates::internal::get_descriptor_hash_code(Any{inclination::Degrees{}}));
  EXPECT_NE(OpenKalman::coordinates::internal::get_descriptor_hash_code(Any{angle::Radians{}}), OpenKalman::coordinates::internal::get_descriptor_hash_code(Any{Polar{}}));

  EXPECT_TRUE(Any{angle::Radians{}} == Any{angle::Radians{}});
  EXPECT_TRUE(Any{angle::Radians{}} != Any{angle::Degrees{}});
  EXPECT_TRUE(Any{inclination::Radians{}} == Any{inclination::Radians{}});
  EXPECT_TRUE(Any{inclination::Radians{}} != Any{inclination::Degrees{}});
  EXPECT_TRUE(Any{angle::Radians{}} != Any{inclination::Radians{}});

  EXPECT_TRUE(test::is_near(to_stat_space(Any{angle::PositiveDegrees{}}, std::array{-390.}), std::array{sqrt3_2, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Any{angle::PositiveDegrees{}}, std::array{-sqrt3_2, -0.5}), std::array{210.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Any{angle::Circle{}}, std::vector{-0.2}), std::array{0.8}, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(Any{inclination::Degrees{}}, std::vector{-60.}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Any{inclination::Degrees{}}, std::vector{-0.5, sqrt3_2}), std::array{120.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Any{inclination::Degrees{}}, std::vector{-380.}), std::array{20.}, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(Any{Polar<Distance, angle::PositiveDegrees>{}}, std::array{-3., -420.}), std::array{3., -0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Any{Polar<Distance, angle::PositiveDegrees>{}}, std::array{3., 0.5, -sqrt3_2}), std::array{3., 300.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Any{Polar<angle::PositiveDegrees, Distance>{}}, std::vector{-10., -2.}), std::array{170., 2.}, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(Any{Spherical<inclination::Degrees, angle::Radians, Distance>{}}, std::vector{-240., pi/6, 2.}), std::array{2., 0.75, sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(Any{Spherical<inclination::Degrees, angle::Radians, Distance>{}}, std::vector{2., 0.75, sqrt3_4, -0.5}), std::array{120., pi/6, 2.}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(Any{Spherical<inclination::Degrees, angle::Radians, Distance>{}}, std::vector{-240., pi/6, 2.}), std::array{120., pi/6, 2.}, 1e-6));
}


TEST(coordinates, descriptor_combinations)
{
  static_assert(std::is_same_v<std::common_type_t<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>, std::size_t>);
  static_assert(std::is_same_v<std::common_type_t<values::Fixed<std::size_t, 2>, std::integral_constant<std::size_t, 3>>, std::size_t>);
  static_assert(std::is_same_v<std::common_type_t<std::integral_constant<std::size_t, 2>, values::Fixed<std::size_t, 3>>, std::size_t>);
  static_assert(std::is_same_v<std::common_type_t<values::Fixed<std::size_t, 2>, values::Fixed<std::size_t, 3>>, std::size_t>);
  static_assert(std::is_same_v<std::common_type_t<std::integral_constant<std::size_t, 3>, std::size_t>, std::size_t>);
  static_assert(std::is_same_v<std::common_type_t<std::size_t, std::integral_constant<std::size_t, 3>>, std::size_t>);

  static_assert(std::is_same_v<std::common_type_t<Dimensions<3>, Dimensions<4>>, Dimensions<>>);
  static_assert(std::is_same_v<std::common_type_t<Dimensions<3>, std::integral_constant<std::size_t, 3>>, Dimensions<3>>);
  static_assert(std::is_same_v<std::common_type_t<Dimensions<3>, Any<double>>, Any<double>>);
  static_assert(std::is_same_v<std::common_type_t<Dimensions<>, unsigned>, Dimensions<>>);
  static_assert(std::is_same_v<std::common_type_t<unsigned, Dimensions<>>, Dimensions<>>);

  static_assert(std::is_same_v<std::common_type_t<Any<float>, Dimensions<3>>, Any<float>>);
  static_assert(std::is_same_v<std::common_type_t<Any<float>, Dimensions<>>, Any<float>>);
  static_assert(std::is_same_v<std::common_type_t<Dimensions<3>, Any<double>>, Any<double>>);
  static_assert(std::is_same_v<std::common_type_t<Dimensions<>, Any<double>>, Any<double>>);

  static_assert(std::is_same_v<std::common_type_t<Distance, Any<float>>, Any<float>>);
  static_assert(std::is_same_v<std::common_type_t<angle::Degrees, Any<double>>, Any<double>>);
  static_assert(std::is_same_v<std::common_type_t<inclination::Degrees, Any<float>>, Any<float>>);
  static_assert(std::is_same_v<std::common_type_t<Polar<>, Any<double>>, Any<double>>);
  static_assert(std::is_same_v<std::common_type_t<Spherical<>, Any<float>>, Any<float>>);

  static_assert(std::is_same_v<std::common_type_t<Any<float>, Distance>, Any<float>>);
  static_assert(std::is_same_v<std::common_type_t<Any<double>, angle::Degrees>, Any<double>>);
  static_assert(std::is_same_v<std::common_type_t<Any<float>, inclination::Degrees>, Any<float>>);
  static_assert(std::is_same_v<std::common_type_t<Any<double>, Polar<>>, Any<double>>);
  static_assert(std::is_same_v<std::common_type_t<Any<float>, Spherical<>>, Any<float>>);

  static_assert(std::is_same_v<std::common_type_t<angle::Degrees, inclination::Radians>, Any<>>);
  static_assert(std::is_same_v<std::common_type_t<Distance, inclination::Radians>, Any<>>);
  static_assert(std::is_same_v<std::common_type_t<angle::Degrees, Polar<>>, Any<>>);
  static_assert(std::is_same_v<std::common_type_t<Spherical<>, inclination::Radians>, Any<>>);
}

