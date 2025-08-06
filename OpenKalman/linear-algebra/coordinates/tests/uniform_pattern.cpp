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
 * \brief Tests relating to \ref uniform_pattern objects
 */

#include "collections/tests/tests.hpp"
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

#include "linear-algebra/coordinates/traits/common_descriptor_type.hpp"

TEST(coordinates, common_descriptor_type)
{
  static_assert(stdcompat::same_as<common_descriptor_type_t<std::tuple<Dimensions<1>>>, Axis>);
  static_assert(stdcompat::same_as<common_descriptor_type_t<std::tuple<Dimensions<10>, Dimensions<1>>>, Axis>);
  static_assert(stdcompat::same_as<common_descriptor_type_t<std::tuple<Axis, Axis>>, Axis>);

  static_assert(stdcompat::same_as<common_descriptor_type_t<std::tuple<Polar<>, Polar<>>>, Polar<>>);
  static_assert(stdcompat::same_as<common_descriptor_type_t<std::array<Polar<>, 10>>, Polar<>>);

  static_assert(stdcompat::same_as<common_descriptor_type_t<std::vector<Polar<>>>, Polar<>>);
  static_assert(stdcompat::same_as<common_descriptor_type_t<stdcompat::ranges::repeat_view<Polar<>>>, Polar<>>);
}

#include "linear-algebra/coordinates/traits/uniform_pattern_type.hpp"

TEST(coordinates, uniform_pattern_type)
{
  static_assert(stdcompat::same_as<uniform_pattern_type_t<Dimensions<1>>, Axis>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<Dimensions<5>>, Axis>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::integral_constant<std::size_t, 5>>, Axis>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<Dimensions<>>, Axis>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::size_t>, Axis>);

  static_assert(stdcompat::same_as<uniform_pattern_type_t<Distance>, Distance>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<angle::Radians>, angle::Radians>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<inclination::Radians>, inclination::Radians>);

  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::tuple<Axis>>, Axis>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::tuple<Axis, Dimensions<4>>>, Axis>);

  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::tuple<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::array<angle::Radians, 10>>, angle::Radians>);

  static_assert(stdcompat::same_as<uniform_pattern_type_t<std::vector<angle::Radians>>, angle::Radians>);
  static_assert(stdcompat::same_as<uniform_pattern_type_t<stdcompat::ranges::repeat_view<angle::Radians>>, angle::Radians>);
}


#include "linear-algebra/coordinates/concepts/uniform_pattern.hpp"

TEST(coordinates, uniform_pattern)
{
  static_assert(uniform_pattern<Dimensions<5>>);
  static_assert(uniform_pattern<Dimensions<0>>);
  static_assert(uniform_pattern<Angle<>>);
  static_assert(not uniform_pattern<Polar<>>);
  static_assert(not uniform_pattern<Spherical<>>);

  static_assert(uniform_pattern<std::tuple<>>);
  static_assert(uniform_pattern<std::tuple<Axis, Axis, Axis>>);
  static_assert(uniform_pattern<std::tuple<Distance, Distance, Distance>>);
  static_assert(uniform_pattern<std::tuple<Angle<>, Angle<>, Angle<>>>);
  static_assert(uniform_pattern<std::tuple<Inclination<>, Inclination<>, Inclination<>>>);

  static_assert(not uniform_pattern<std::tuple<Axis, Distance, Angle<>, Inclination<>>>);
  static_assert(not uniform_pattern<std::tuple<Polar<>, Polar<>>>);
  static_assert(not uniform_pattern<std::tuple<Axis, angle::Radians>>);
}


#include "linear-algebra/coordinates/functions/is_uniform_pattern_component_of.hpp"

TEST(coordinates, is_uniform_pattern_component_of)
{
  // fixed:
  static_assert(is_uniform_pattern_component_of(Axis{}, Axis{}));
  static_assert(is_uniform_pattern_component_of(Axis{}, Dimensions<10>{}));
  static_assert(not is_uniform_pattern_component_of(Dimensions<2>{}, Dimensions<10>{}));
  static_assert(not is_uniform_pattern_component_of(Distance{}, Dimensions<10>{}));
  static_assert(not is_uniform_pattern_component_of(Axis{}, std::tuple<Dimensions<10>, Distance>{}));
  static_assert(is_uniform_pattern_component_of(Distance{}, std::tuple<Distance, Distance, Distance, Distance>{}));
  static_assert(is_uniform_pattern_component_of(angle::Radians{}, std::tuple<angle::Radians, angle::Radians, angle::Radians, angle::Radians>{}));
  static_assert(not is_uniform_pattern_component_of(Polar<>{}, std::array<Polar<>, 4>{}));

  // dynamic:
  static_assert(is_uniform_pattern_component_of(1U, Dimensions<10>{}));
  static_assert(is_uniform_pattern_component_of(Dimensions<1>{}, 10U));
  static_assert(is_uniform_pattern_component_of(1U, 10U));
  static_assert(not is_uniform_pattern_component_of(2U, 10U));
  static_assert(is_uniform_pattern_component_of(Dimensions{1}, Dimensions{5}));
  static_assert(not is_uniform_pattern_component_of(Dimensions{2}, Dimensions{5}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Axis{}, std::vector {Axis{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Axis{}, std::vector {Dimensions<10>{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Distance{}}, std::array<Distance, 4>{}));

  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Axis{}}, Dimensions<5>{}));
  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Axis{}}, std::vector{Dimensions<2>{}, Dimensions<2>{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Any{Axis{}}}, std::vector{Dimensions<2>{}, Dimensions<2>{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Any{Axis{}}}, std::vector{Any{Dimensions<2>{}}, Any{Dimensions<2>{}}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Any{Dimensions<0>{}}, Any{Distance{}}, Any{Dimensions<0>{}}}, std::vector{Distance{}, Distance{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Any{Dimensions<0>{}}, Any{Distance{}}, Any{Dimensions<0>{}}}, std::vector{Any{Distance{}}, Any{Dimensions<0>{}}, Any{Distance{}}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(std::vector{Any{Axis{}}, Any{Dimensions<0>{}}}, Any{Dimensions<2>{}}));
  EXPECT_FALSE(is_uniform_pattern_component_of(std::vector{Any{Axis{}}, Any{Dimensions<0>{}}}, Any{Dimensions<0>{}}));

  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Axis{}}, Any{Axis{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Axis{}}, Any{Dimensions<10>{}}));
  EXPECT_FALSE(is_uniform_pattern_component_of(Any{Dimensions<2>{}}, Any{Dimensions<10>{}}));
  EXPECT_FALSE(is_uniform_pattern_component_of(Any{Dimensions<1>{}}, Any{Dimensions<0>{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Axis{}}, Any{Dimensions<2>{}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Distance{}}, Any{Distance{}}));

  EXPECT_TRUE(is_uniform_pattern_component_of(Any<double>{Axis{}}, Any<float>{Dimensions<2>{}}));

  static_assert(is_uniform_pattern_component_of(Dimensions<1>{}, stdcompat::ranges::views::repeat(Dimensions<1>{})));
  static_assert(not is_uniform_pattern_component_of(stdcompat::ranges::views::repeat(Dimensions<1>{}), Dimensions<100>{}));
  static_assert(not is_uniform_pattern_component_of(Dimensions<1>{}, stdcompat::ranges::views::empty<Dimensions<1>>));
  static_assert(not is_uniform_pattern_component_of(stdcompat::ranges::views::empty<Dimensions<1>>, Dimensions<10>{}));
  static_assert(is_uniform_pattern_component_of(Dimensions<1>{}, stdcompat::ranges::views::single(Dimensions<1>{})));
  static_assert(is_uniform_pattern_component_of(stdcompat::ranges::views::single(Dimensions<1>{}), Dimensions<100>{}));

  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Axis{}}, std::vector{Any{Dimensions<2>{}}, Any{Dimensions<3>{}}}));
  EXPECT_FALSE(is_uniform_pattern_component_of(Any{Dimensions<2>{}}, std::vector{Any{Dimensions<2>{}}, Any{Dimensions<3>{}}}));
  EXPECT_FALSE(is_uniform_pattern_component_of(Any{Axis{}}, std::vector{Any{Dimensions<2>{}}, Any{Distance{}}}));
  EXPECT_TRUE(is_uniform_pattern_component_of(Any{Distance{}}, std::vector{Distance{}, Distance{}, Distance{}, Distance{}}));
}


#include "linear-algebra/coordinates/functions/get_uniform_pattern_component.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"

TEST(coordinates, get_uniform_pattern_component)
{
  static_assert(stdcompat::is_eq(compare(get_uniform_pattern_component(Dimensions<10>{}).value(), Axis{})));
  static_assert(stdcompat::is_eq(compare(get_uniform_pattern_component(std::tuple<Distance, Distance, Distance>{}).value(), Distance{})));
  static_assert(stdcompat::is_eq(compare(get_uniform_pattern_component(std::tuple{Dimensions<5>{}, Dimensions{5}, Dimensions{5}}).value(), Dimensions<1>{})));
  EXPECT_TRUE(not get_uniform_pattern_component(std::tuple<Distance, Distance, Axis>{}));
  EXPECT_TRUE(not get_uniform_pattern_component(std::tuple{Distance{}, Distance{}, Dimensions{5}}));
  EXPECT_TRUE(stdcompat::is_eq(compare(get_uniform_pattern_component(std::tuple{Any{Distance{}}, Any{Distance{}}, Any{Distance{}}}).value(), Distance{})));
  EXPECT_TRUE(stdcompat::is_eq(compare(get_uniform_pattern_component(std::vector{Any{Distance{}}, Any{Distance{}}, Any{Distance{}}}).value(), Distance{})));
  EXPECT_TRUE(not get_uniform_pattern_component(std::tuple{Any{Distance{}}, Any{Distance{}}, Any{Angle{}}}));
  EXPECT_TRUE(not get_uniform_pattern_component(std::vector{Any{Distance{}}, Any{Distance{}}, Any{Angle{}}}));

  static_assert(stdcompat::is_eq(compare(get_uniform_pattern_component(stdcompat::ranges::views::repeat(Dimensions<1>{})).value(), Dimensions<1>{})));
  static_assert(stdcompat::is_eq(compare(get_uniform_pattern_component(stdcompat::ranges::views::empty<Dimensions<1>>).value(), Dimensions<1>{})));
  static_assert(stdcompat::is_eq(compare(get_uniform_pattern_component(std::tuple{}).value(), Dimensions<1>{})));
}
