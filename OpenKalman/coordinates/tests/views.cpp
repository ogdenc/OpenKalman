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
  EXPECT_TRUE(compare(concat(stdcompat::cref(d0), std::tuple{d0, Dimensions<0>{}}), std::tuple<Distance, Distance>{}));
  EXPECT_TRUE(compare(concat(std::tuple{Dimensions<0>{}, d0}, stdcompat::cref(d0)), std::tuple<Distance, Distance>{}));
  EXPECT_TRUE(compare(concat(stdcompat::cref(d0), stdcompat::cref(d0)), std::tuple<Distance, Distance>{}));
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

/*#include "coordinates/views/repeat.hpp"
#include "coordinates/views/replicate.hpp"

TEST(coordinates, fixed_replication)
{
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 4>{} == Dimensions<12>{});
  static_assert(std::integral_constant<std::size_t, 4>{} * Dimensions<3>{} == Dimensions<12>{});
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 4>{} * std::integral_constant<std::size_t, 2>{} == Dimensions<24>{});
  static_assert(collections::views::all(std::tuple{}) * std::integral_constant<std::size_t, 4>{} == Dimensions<0>{});
  static_assert(std::integral_constant<std::size_t, 4>{} * collections::views::all(std::tuple{}) == Dimensions<0>{});
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 0>{} == Dimensions<0>{});
  static_assert(std::integral_constant<std::size_t, 0>{} * Dimensions<3>{} == Dimensions<0>{});

  static_assert(compare(Distance{} * std::integral_constant<std::size_t, 1>{}, Distance{}));
  static_assert(compare(std::integral_constant<std::size_t, 1>{} * Distance{}, Distance{}));
  static_assert(compare(Distance{} * std::integral_constant<std::size_t, 2>{}, std::array<Distance, 2>{}));
  static_assert(compare(std::integral_constant<std::size_t, 2>{} * Distance{}, std::array<Distance, 2>{}));
  static_assert(compare(std::tuple<Distance, angle::Radians>{} * std::integral_constant<std::size_t, 2>{}, std::tuple<Distance, angle::Radians, Distance, angle::Radians>{}));

  static_assert(compare(std::tuple<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 1>{}, std::tuple<Axis, Distance, Axis>{}));
  static_assert(compare(std::tuple<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 2>{}, std::tuple<Axis, Distance, Dimensions<2>, Distance, Axis>{}));
  static_assert(compare(std::tuple<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 3>{}, std::tuple<Axis, Distance, Dimensions<2>, Distance, Dimensions<2>, Distance, Axis>{}));
  static_assert(compare(std::tuple<Axis, Distance, Dimensions<2>>{} * std::integral_constant<std::size_t, 4>{}, std::tuple<Axis, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<2>>{}));
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

  EXPECT_TRUE(compare(std::vector {Any{Axis{}}, Any{Distance{}}} * std::integral_constant<std::size_t, 2>{}, std::vector {Any{Axis{}}, Any{Distance{}}, Any{Axis{}}, Any{Distance{}}}));
  EXPECT_TRUE(compare(std::vector {Any{Axis{}}, Any{Distance{}}} * 2u, std::vector {Any{Axis{}}, Any{Distance{}}, Any{Axis{}}, Any{Distance{}}}));
  EXPECT_TRUE(compare(std::vector {Any{Axis{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Axis{}}} * 2u, std::vector {Any{Axis{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Dimensions<2>{}}, Any{Distance{}}, Any{angle::Degrees{}}, Any{Axis{}}}));
}
*/