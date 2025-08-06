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
 * \brief Tests for \ref coordinates::pattern arithmetic.
 */

#include "collections/tests/tests.hpp"
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/functions/arithmetic-operators.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

TEST(coordinates, fixed_concatenation)
{
  static_assert(Dimensions<3>{} + Dimensions<4>{} == Dimensions<7>{});
  static_assert(std::tuple<Axis, Axis>{} + std::tuple<Axis, Axis, Axis>{} == Dimensions<5>{});
  static_assert(Polar{} + Dimensions<2>{} == std::tuple<Polar<>, Dimensions<2>>{});

  static_assert(collections::views::all(std::tuple<>{}) + std::tuple<>{} == Dimensions<0>{});
  static_assert(std::tuple<Axis>{} + std::tuple<>{} == Axis{});
  static_assert(std::tuple<>{} + std::tuple<Axis>{} == Axis{});
  static_assert(Axis{} + std::tuple<angle::Radians>{} == std::tuple<Axis, angle::Radians>{});
  static_assert(std::tuple<Axis>{} + std::tuple<angle::Radians>{} == std::tuple<Axis, angle::Radians>{});
  static_assert(std::tuple<Axis, angle::Radians>{} + std::tuple<angle::Radians, Axis>{} == std::tuple<Axis, angle::Radians, angle::Radians, Axis>{});
  static_assert(std::tuple<Axis, angle::Radians>{} + std::tuple<angle::Radians, Axis>{} + std::tuple<Axis, angle::Radians>{} ==
    std::tuple<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>{});
  static_assert(Axis{} + Polar{} == std::tuple<Axis, Polar<>>{});
  static_assert(Polar{} + Axis{} == std::tuple<Polar<>, Axis>{});
  static_assert(Polar{} + Spherical<>{} + Polar<>{} == std::tuple<Polar<>, Spherical<>, Polar<>>{});

  static_assert(stdcompat::is_eq(compare(std::tuple<Dimensions<0>, Distance, Dimensions<2>>{} + std::tuple<Axis, Dimensions<2>, inclination::Radians>{},
    std::tuple<Distance, Dimensions<5>, inclination::Radians>{})));
  static_assert(stdcompat::is_eq(compare(std::tuple<Dimensions<0>, Distance, Dimensions<0>>{} + std::tuple<Dimensions<0>, inclination::Radians>{},
    std::tuple<Distance, inclination::Radians>{})));
  static_assert(stdcompat::is_eq(compare(Dimensions<0>{} + inclination::Radians{}, std::tuple<inclination::Radians>{})));
  static_assert(stdcompat::is_eq(compare(inclination::Radians{} + Dimensions<0>{}, std::tuple<inclination::Radians>{})));

  static constexpr auto d0 = Distance{};
  static_assert(stdcompat::is_eq(compare(std::tuple{Dimensions<0>{}, d0} + std::tuple{d0, Dimensions<0>{}}, std::tuple<Distance, Distance>{})));
}


TEST(coordinates, dynamic_concatenation)
{
  static_assert(Dimensions{3} + Dimensions{4} == Dimensions{7});

  auto ca = std::vector {Any{Axis{}}, Any{angle::Radians{}}} + std::vector {Any{Dimensions<3>{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}};
  auto ita = std::begin(ca);
  static_assert(std::is_same_v<decltype(*ita), Any<>>);
  EXPECT_EQ(coordinates::internal::get_descriptor_hash_code(*ita), coordinates::internal::get_descriptor_hash_code(Any<>{Axis{}}));
  EXPECT_TRUE(*ita == Axis{});
  EXPECT_TRUE(*(ita + 1) == angle::Radians{});
  EXPECT_TRUE(ita[1] == angle::Radians{});
  EXPECT_TRUE(*(2 + ita) == Dimensions<3>{});
  EXPECT_TRUE(ita[3] == angle::Degrees{});
  ++ita;
  EXPECT_TRUE(*ita == angle::Radians{});
  EXPECT_TRUE(*(ita - 1) == Axis{});
  EXPECT_TRUE(ita[-1] == Axis{});
  EXPECT_TRUE(ita[1] == Dimensions<3>{});
  EXPECT_TRUE(ita++[2] == angle::Degrees{});
  EXPECT_TRUE(*ita == Dimensions<3>{});
  EXPECT_TRUE(*(ita - 2) == Axis{});
  EXPECT_TRUE(ita - ++std::begin(ca) == 1);
  EXPECT_TRUE(ita[1] == angle::Degrees{});
  EXPECT_TRUE(ita--[2] == Dimensions<2>{});
  EXPECT_TRUE(*ita == angle::Radians{});
  --ita;
  EXPECT_TRUE(*ita == Axis{});

  EXPECT_TRUE((std::vector {Any{Axis{}}, Any{angle::Radians{}}} + std::vector {Any{angle::Degrees{}}, Any{Dimensions<2>{}}} ==
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}}));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{Dimensions<2>{}}} + std::vector {Any{Axis{}}, Any{angle::Degrees{}}, Any{Dimensions<4>{}}},
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{Dimensions<3>{}}, Any{angle::Degrees{}}, Any{Dimensions<4>{}}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::vector {Any{Axis{}}, Any{angle::Radians{}}} + Dimensions<2>{},
    std::tuple {Axis{}, angle::Radians{}, Dimensions<2>{}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(Dimensions<2>{} + std::tuple {angle::Degrees{}, Axis{}},
    std::tuple {Dimensions<2>{}, angle::Degrees{}, Axis{}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::tuple {Axis{}, angle::Radians{}} + angle::Degrees{},
    std::tuple {Axis{}, angle::Radians{}, angle::Degrees{}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(angle::Radians{} + std::tuple {angle::Degrees{}, Axis{}},
    std::tuple {angle::Radians{}, angle::Degrees{}, Axis{}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::vector {Any{Axis{}}, Any{angle::Radians{}}} + std::tuple<angle::Degrees, Axis>{},
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{angle::Degrees{}}, Any{Axis{}}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::tuple<Axis, angle::Radians>{} + std::vector {Any{angle::Degrees{}}, Any{Axis{}}},
    std::vector {Any{Axis{}}, Any{angle::Radians{}}, Any{angle::Degrees{}}, Any{Axis{}}})));
}


TEST(coordinates, fixed_replication)
{
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 4>{} == Dimensions<12>{});
  static_assert(std::integral_constant<std::size_t, 4>{} * Dimensions<3>{} == Dimensions<12>{});
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 4>{} * std::integral_constant<std::size_t, 2>{} == Dimensions<24>{});
  static_assert(collections::views::all(std::tuple{}) * std::integral_constant<std::size_t, 4>{} == Dimensions<0>{});
  static_assert(std::integral_constant<std::size_t, 4>{} * collections::views::all(std::tuple{}) == Dimensions<0>{});
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 0>{} == Dimensions<0>{});
  static_assert(std::integral_constant<std::size_t, 0>{} * Dimensions<3>{} == Dimensions<0>{});

  static_assert(stdcompat::is_eq(compare(Distance{} * std::integral_constant<std::size_t, 1>{}, Distance{})));
  static_assert(stdcompat::is_eq(compare(std::integral_constant<std::size_t, 1>{} * Distance{}, Distance{})));
  static_assert(stdcompat::is_eq(compare(Distance{} * std::integral_constant<std::size_t, 2>{}, std::array<Distance, 2>{})));
  static_assert(stdcompat::is_eq(compare(std::integral_constant<std::size_t, 2>{} * Distance{}, std::array<Distance, 2>{})));
  static_assert(stdcompat::is_eq(compare(std::tuple<Distance, angle::Radians>{} * std::integral_constant<std::size_t, 2>{}, std::tuple<Distance, angle::Radians, Distance, angle::Radians>{})));

  static_assert(stdcompat::is_eq(compare(std::tuple<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 1>{}, std::tuple<Axis, Distance, Axis>{})));
  static_assert(stdcompat::is_eq(compare(std::tuple<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 2>{}, std::tuple<Axis, Distance, Dimensions<2>, Distance, Axis>{})));
  static_assert(stdcompat::is_eq(compare(std::tuple<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 3>{}, std::tuple<Axis, Distance, Dimensions<2>, Distance, Dimensions<2>, Distance, Axis>{})));
  static_assert(stdcompat::is_eq(compare(std::tuple<Axis, Distance, Dimensions<2>>{} * std::integral_constant<std::size_t, 4>{}, std::tuple<Axis, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<2>>{})));
}


TEST(coordinates, dynamic_replication)
{
  static_assert(Dimensions<3>{} * 4u == Dimensions{12});
  static_assert(4u * Dimensions<3>{} == Dimensions{12});
  static_assert(Dimensions<3>{} * 4u * std::integral_constant<std::size_t, 2>{} == Dimensions{24});
  static_assert(collections::views::all(std::tuple<>{}) * 4u == Dimensions{0});
  static_assert(4u * collections::views::all(std::tuple<>{}) == Dimensions{0});
  static_assert(Dimensions<3>{} * 0u == Dimensions{0});
  static_assert(0u * Dimensions<3>{} == Dimensions{0});

  static_assert(Dimensions{3} * 4u == Dimensions{12});
  static_assert(4u * Dimensions{3} == Dimensions{12});
  static_assert(Dimensions{3} * 4u * std::integral_constant<std::size_t, 2>{} == Dimensions{24});
  static_assert(Dimensions{0} * 4u == Dimensions{0});
  static_assert(4u * Dimensions{0} == Dimensions{0});
  static_assert(Dimensions{3} * 0u == Dimensions{0});
  static_assert(0u * Dimensions{3} == Dimensions{0});

  EXPECT_TRUE(stdcompat::is_eq(compare(std::vector {Any{Axis{}}, Any{Distance{}}} * std::integral_constant<std::size_t, 2>{}, std::vector {Any{Axis{}}, Any{Distance{}}, Any{Axis{}}, Any{Distance{}}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::vector {Any{Axis{}}, Any{Distance{}}} * 2u, std::vector {Any{Axis{}}, Any{Distance{}}, Any{Axis{}}, Any{Distance{}}})));
  EXPECT_TRUE(stdcompat::is_eq(compare(std::vector {Any{Axis{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Axis{}}} * 2u, std::vector {Any{Axis{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Axis{}}})));
}