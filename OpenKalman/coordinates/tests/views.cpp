/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref coordinates::pattern views.
 */

#include "collections/tests/tests.hpp"
#include "collections/collections.hpp"
#include "coordinates/functions/compare.hpp"
#include "coordinates/descriptors/Any.hpp"
#include "coordinates/descriptors/Dimensions.hpp"
#include "coordinates/descriptors/Distance.hpp"
#include "coordinates/descriptors/Angle.hpp"
#include "coordinates/descriptors/Inclination.hpp"
#include "coordinates/descriptors/Polar.hpp"
#include "coordinates/descriptors/Spherical.hpp"
#include "coordinates/functions/compare.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

#include "coordinates/views/concat.hpp"

TEST(coordinates, fixed_concatenation)
{
  using coordinates::views::concat;
  static_assert(compare(concat(Dimensions<3>{}, Dimensions<4>{}), Dimensions<7>{}));
  static_assert(compare(concat(std::tuple<Axis, Axis>{}, std::tuple<Axis, Axis, Axis>{}), Dimensions<5>{}));
  static_assert(compare(concat(Polar{}, Dimensions<2>{}), std::tuple<Polar<>, Dimensions<2>>{}));

  static_assert(compare(concat(collections::views::all(std::tuple<>{}), std::tuple<>{}), Dimensions<0>{}));
  static_assert(compare(concat(std::tuple<Axis>{}, std::tuple<>{}), Axis{}));
  static_assert(compare(concat(std::tuple<>{}, std::tuple<Axis>{}), Axis{}));
  static_assert(compare(concat(Axis{}, angle::Radians{}), std::tuple<Axis, angle::Radians>{}));
  static_assert(compare(concat(Axis{}, std::tuple<angle::Radians>{}), std::tuple<Axis, angle::Radians>{}));
  static_assert(compare(concat(std::tuple<Axis>{}, std::tuple<angle::Radians>{}), std::tuple<Axis, angle::Radians>{}));
  static_assert(compare(concat(std::tuple<Axis, angle::Radians>{}, std::tuple<angle::Radians, Axis>{}), std::tuple<Axis, angle::Radians, angle::Radians, Axis>{}));
  static_assert(compare(concat(std::tuple<Axis, angle::Radians>{}, std::tuple<angle::Radians, Axis>{}, std::tuple<Axis, angle::Radians>{}),
    std::tuple<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>{}));
  static_assert(compare(concat(Axis{}, Polar{}), std::tuple<Axis, Polar<>>{}));
  static_assert(compare(concat(Polar{}, Axis{}), std::tuple<Polar<>, Axis>{}));
  static_assert(compare(concat(Polar{}, Spherical<>{}, Polar<>{}), std::tuple<Polar<>, Spherical<>, Polar<>>{}));

  static_assert(compare(concat(std::tuple<Dimensions<0>, Distance, Dimensions<2>>{}, std::tuple<Axis, Dimensions<2>, inclination::Radians>{}),
    std::tuple<Distance, Dimensions<5>, inclination::Radians>{}));
  static_assert(compare(concat(std::tuple<Dimensions<0>, Distance, Dimensions<0>>{}, std::tuple<Dimensions<0>, inclination::Radians>{}),
    std::tuple<Distance, inclination::Radians>{}));
  static_assert(compare(concat(Dimensions<0>{}, inclination::Radians{}), std::tuple<inclination::Radians>{}));
  static_assert(compare(concat(inclination::Radians{}, Dimensions<0>{}), std::tuple<inclination::Radians>{}));

  static constexpr auto d0 = Distance{};
  EXPECT_TRUE(compare(concat(std::tuple{Dimensions<0>{}, d0}, std::tuple{d0, Dimensions<0>{}}), std::tuple<Distance, Distance>{}));
  EXPECT_TRUE(compare(concat(stdex::cref(d0), std::tuple{d0, Dimensions<0>{}}), std::tuple<Distance, Distance>{}));
  EXPECT_TRUE(compare(concat(std::tuple{Dimensions<0>{}, d0}, stdex::cref(d0)), std::tuple<Distance, Distance>{}));
  EXPECT_TRUE(compare(concat(stdex::cref(d0), stdex::cref(d0)), std::tuple<Distance, Distance>{}));
}


TEST(coordinates, dynamic_concatenation)
{
  using coordinates::views::concat;
  static_assert(compare(concat(Dimensions{3}, Dimensions{4}), Dimensions{7}));

  auto ca = concat(std::vector {Any{Axis{}}, Any{angle::Radians{}}}, std::vector {Any{Dimensions<3>{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}});
  auto ita = std::begin(ca);
  static_assert(std::is_same_v<decltype(*ita), Any<>>);
  EXPECT_EQ(coordinates::internal::get_descriptor_hash_code(*ita), coordinates::internal::get_descriptor_hash_code(Any<>{Axis{}}));
  EXPECT_TRUE(compare(*ita, Axis{}));
  EXPECT_TRUE(compare(*(ita + 1), angle::Radians{}));
  EXPECT_TRUE(compare(ita[1], angle::Radians{}));
  EXPECT_TRUE(compare(*(2 + ita), Dimensions<3>{}));
  EXPECT_TRUE(compare(ita[3], angle::Degrees{}));
  ++ita;
  EXPECT_TRUE(compare(*ita, angle::Radians{}));
  EXPECT_TRUE(compare(*(ita - 1), Axis{}));
  EXPECT_TRUE(compare(ita[-1], Axis{}));
  EXPECT_TRUE(compare(ita[1], Dimensions<3>{}));
  EXPECT_TRUE(compare(ita++[2], angle::Degrees{}));
  EXPECT_TRUE(compare(*ita, Dimensions<3>{}));
  EXPECT_TRUE(compare(*(ita - 2), Axis{}));
  EXPECT_EQ(ita - ++std::begin(ca), 1);
  EXPECT_TRUE(compare(ita[1], angle::Degrees{}));
  EXPECT_TRUE(compare(ita--[2], Dimensions<2>{}));
  EXPECT_TRUE(compare(*ita, angle::Radians{}));
  --ita;
  EXPECT_TRUE(compare(*ita, Axis{}));

  EXPECT_TRUE(compare(concat(std::vector {Any{Axis{}}, Any{angle::Radians{}}}, std::vector {Any{angle::Degrees{}}, Any{Dimensions<2>{}}}),
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}}));
  EXPECT_TRUE(compare(concat(std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{Dimensions<2>{}}}, std::vector {Any{Axis{}}, Any{angle::Degrees{}}, Any{Dimensions<4>{}}}),
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{Dimensions<3>{}}, Any{angle::Degrees{}}, Any{Dimensions<4>{}}}));
  EXPECT_TRUE(compare(concat(std::vector {Any{Axis{}}, Any{angle::Radians{}}}, Dimensions<2>{}),
    std::tuple {Axis{}, angle::Radians{}, Dimensions<2>{}}));
  EXPECT_TRUE(compare(concat(Dimensions<2>{}, std::tuple {angle::Degrees{}, Axis{}}),
    std::tuple {Dimensions<2>{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE(compare(concat(std::tuple {Axis{}, angle::Radians{}}, angle::Degrees{}),
    std::tuple {Axis{}, angle::Radians{}, angle::Degrees{}}));
  EXPECT_TRUE(compare(concat(angle::Radians{}, std::tuple {angle::Degrees{}, Axis{}}),
    std::tuple {angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE(compare(concat(std::vector {Any{Axis{}}, Any{angle::Radians{}}}, std::tuple<angle::Degrees, Axis>{}),
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{angle::Degrees{}}, Any{Axis{}}}));
  EXPECT_TRUE(compare(concat(std::tuple<Axis, angle::Radians>{}, std::vector {Any{angle::Degrees{}}, Any{Axis{}}}),
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{angle::Degrees{}}, Any{Axis{}}}));
}

#include "coordinates/views/replicate.hpp"

TEST(coordinates, replicate)
{
  using coordinates::views::replicate;

  static_assert(coordinates::dimension_of_v<decltype(replicate(Dimensions<3>{}, std::integral_constant<std::size_t, 4>{}))> == 12);
  static_assert(coordinates::get_dimension(replicate(Dimensions<3>{}, 4u))== 12);
  static_assert(coordinates::get_dimension(replicate(Dimensions{3}, std::integral_constant<std::size_t, 4>{})) == 12);
  static_assert(coordinates::get_dimension(replicate(Dimensions{3}, 4_uz)) == 12);
  static_assert(coordinates::get_dimension(replicate(Dimensions{3}, 4u)) == 12);

  static_assert(compare(replicate(Dimensions<3>{}, std::integral_constant<std::size_t, 4>{}), Dimensions<12>{}));
  static_assert(compare(replicate(Dimensions<3>{}, std::integral_constant<std::size_t, 8>{}), Dimensions<24>{}));
  static_assert(compare(replicate(collections::views::all(std::tuple{}), std::integral_constant<std::size_t, 4>{}), Dimensions<0>{}));
  static_assert(compare(replicate(Dimensions<3>{}, std::integral_constant<std::size_t, 0>{}), Dimensions<0>{}));

  EXPECT_TRUE(compare(replicate(Distance{}, std::integral_constant<std::size_t, 1>{}), Distance{}));
  static_assert(compare(replicate(Distance{}, std::integral_constant<std::size_t, 2>{}), std::array<Distance, 2>{}));
  static_assert(compare(replicate(std::tuple<Distance, angle::Radians>{}, std::integral_constant<std::size_t, 2>{}), std::tuple<Distance, angle::Radians, Distance, angle::Radians>{}));

  static_assert(compare(replicate(std::tuple<Axis, Distance, Axis>{}, std::integral_constant<std::size_t, 1>{}), std::tuple<Axis, Distance, Axis>{}));
  static_assert(compare(replicate(std::tuple<Axis, Distance, Axis>{}, std::integral_constant<std::size_t, 2>{}), std::tuple<Axis, Distance, Dimensions<2>, Distance, Axis>{}));
  static_assert(compare(replicate(std::tuple<Axis, Distance, Axis>{}, std::integral_constant<std::size_t, 3>{}), std::tuple<Axis, Distance, Dimensions<2>, Distance, Dimensions<2>, Distance, Axis>{}));
  static_assert(compare(replicate(std::tuple<Axis, Distance, Dimensions<2>>{}, std::integral_constant<std::size_t, 4>{}), std::tuple<Axis, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<2>>{}));

  static_assert(compare(replicate(Dimensions<3>{}, 4u), Dimensions{12}));
  static_assert(compare(replicate(Dimensions<3>{}, 8u), Dimensions{24}));
  static_assert(compare(replicate(collections::views::all(std::tuple<>{}), 4u), Dimensions{0}));
  static_assert(compare(replicate(Dimensions<3>{}, 0u), Dimensions{0}));

  static_assert(compare(replicate(Dimensions{3}, 4u), Dimensions{12}));
  static_assert(compare(replicate(Dimensions{3}, 8u), Dimensions{24}));
  static_assert(compare(replicate(Dimensions{0}, 4u), Dimensions{0}));
  static_assert(compare(replicate(Dimensions{3}, 0u), Dimensions{0}));

  EXPECT_TRUE(compare(replicate(std::vector {Any{Axis{}}, Any{Distance{}}}, std::integral_constant<std::size_t, 2>{}), std::vector {Any{Axis{}}, Any{Distance{}}, Any{Axis{}}, Any{Distance{}}}));
  EXPECT_TRUE(compare(replicate(std::vector {Any{Axis{}}, Any{Distance{}}}, 2u), std::vector {Any{Axis{}}, Any{Distance{}}, Any{Axis{}}, Any{Distance{}}}));
  EXPECT_TRUE(compare(replicate(std::vector {Any{Axis{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Axis{}}}, 2u), std::vector {Any{Axis{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Axis{}}}));
}

#include "coordinates/views/dimensions.hpp"

TEST(coordinates, views_dimensions)
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
  using N4 = std::integral_constant<std::size_t, 4>;
  using coordinates::views::dimensions;
  using collections::compare_indices;

  static_assert(collections::get_element(dimensions(std::tuple{Dimensions<3>{}}), N0{}) == 3);
  static_assert(collections::get_element(dimensions(std::tuple{Dimensions<4>{}}), 0U) == 4);
  static_assert(collections::get_element(dimensions(std::tuple{Dimensions<5>{}, Dimensions<6>{}}), N0{}) == 5);
  static_assert(collections::get_element(dimensions(std::tuple{Dimensions<5>{}, Dimensions<6>{}}), N1{}) == 6);
  static_assert(collections::get_element(dimensions(std::tuple{Dimensions<5>{}, Dimensions<6>{}}), 0U) == 5);
  static_assert(collections::get_element(dimensions(std::tuple{Dimensions<5>{}, Dimensions<6>{}}), 1U) == 6);

  static_assert(collections::get_element(dimensions(std::array{3U}), N0{}) == 3);
  static_assert(collections::get_element(dimensions(std::array{4U}), 0U) == 4);
  static_assert(collections::get_element(dimensions(std::array{5U, 6U}), N0{}) == 5);
  static_assert(collections::get_element(dimensions(std::array{5U, 6U}), N1{}) == 6);
  static_assert(collections::get_element(dimensions(std::array{5U, 6U}), 0U) == 5);
  static_assert(collections::get_element(dimensions(std::array{5U, 6U}), 1U) == 6);

  static_assert(compare_indices(dimensions(std::tuple{Dimensions<3>{}}), std::tuple{N3{}}));
  static_assert(compare_indices(dimensions(std::tuple{Dimensions<3>{}}), std::array{3U}));
  EXPECT_TRUE(compare_indices(dimensions(std::tuple{Dimensions<3>{}}), std::vector{3U}));
  static_assert(compare_indices(dimensions(std::tuple{Dimensions<3>{}, Dimensions<4>{}}), std::tuple{N3{}, N4{}}));
  static_assert(compare_indices<&stdex::is_neq>(dimensions(std::tuple{Dimensions<2>{}, Dimensions<5>{}}), std::tuple{N3{}, N4{}}));
  static_assert(compare_indices(dimensions(collections::views::all(std::tuple{})), std::tuple{}));
  static_assert(compare_indices(dimensions(std::tuple{Angle{}}), std::tuple{N1{}}));
  static_assert(compare_indices(dimensions(std::tuple{Angle{}, Inclination{}, Polar{}}), std::tuple{N1{}, N1{}, N2{}}));
  EXPECT_TRUE(compare_indices(dimensions(std::tuple{Angle{}, Inclination{}, Polar{}}), std::vector{1U, 1U, 2U}));

  static_assert(compare_indices(dimensions(std::tuple{Dimensions<3>{}}), std::tuple{N3{}, N1{}}));
  static_assert(compare_indices(std::tuple{N2{}, N1{}, N1{}}, dimensions(std::tuple{Polar{}})));
  static_assert(compare_indices<&stdex::is_lt>(std::tuple{N1{}, N2{}, N0{}, N0{}}, dimensions(std::tuple{Polar{}, Spherical{}, Inclination{}})));
  static_assert(compare_indices<&stdex::is_lt>(std::tuple{1U, 2U, 0U, 0U}, dimensions(std::tuple{Polar{}, Spherical{}, Inclination{}})));
  static_assert(not compare_indices<&stdex::is_lt>(std::tuple{1U, 3U, 0U, 0U}, dimensions(std::tuple{Polar{}, Spherical{}, Inclination{}})));
  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{1U, 2U, 0U, 0U}, dimensions(std::tuple{Polar{}, Spherical{}, Inclination{}})));
  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{1U, 2U, 0U, 0U}, dimensions(std::vector{Any{Polar{}}, Any{Spherical{}}, Any{Inclination{}}})));

  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{2U, 2U, 0U, 0U}, dimensions(std::tuple{std::tuple{Polar{}, Distance{}}, Any{Spherical{}}, Any{Inclination{}}})));
}
