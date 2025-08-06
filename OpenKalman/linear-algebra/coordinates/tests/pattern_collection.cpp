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
 * \brief Tests relating to \ref pattern_collection objects
 */

#include "collections/tests/tests.hpp"
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
}

#include "linear-algebra/coordinates/concepts/pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern_collection.hpp"

TEST(coordinates, pattern_collection)
{
  static_assert(collections::collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(collections::collection<std::vector<angle::Radians>>);

  static_assert(pattern_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(pattern_tuple<std::array<Distance, 5>>);
  static_assert(not pattern_tuple<std::tuple<Axis, Distance, double, angle::Radians>>);
  static_assert(not pattern_tuple<std::vector<angle::Radians>>);
  static_assert(not pattern_tuple<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_pattern_tuple<std::tuple<Axis, Dimensions<3>, unsigned, std::integral_constant<std::size_t, 5>>>);
  static_assert(not euclidean_pattern_tuple<std::tuple<Axis, Distance, angle::Radians>>);

  static_assert(pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(pattern_collection<std::array<Distance, 5>>);
  static_assert(pattern_collection<std::vector<angle::Radians>>);
  static_assert(pattern_collection<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_pattern_collection<std::tuple<Axis, Dimensions<3>, unsigned, std::integral_constant<std::size_t, 5>>>);
  static_assert(euclidean_pattern_collection<std::array<Dimensions<4>, 5>>);
  static_assert(euclidean_pattern_collection<std::vector<Axis>>);
  static_assert(euclidean_pattern_collection<std::initializer_list<unsigned>>);
  static_assert(not euclidean_pattern_collection<std::array<Distance, 5>>);
  static_assert(not euclidean_pattern_collection<std::vector<angle::Radians>>);
  static_assert(not euclidean_pattern_collection<std::initializer_list<Distance>>);

  static_assert(fixed_pattern_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(not fixed_pattern_tuple<std::tuple<Axis, Distance, unsigned, angle::Radians>>);

  static_assert(fixed_pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(fixed_pattern_collection<std::array<Distance, 5>>);
  static_assert(fixed_pattern_collection<std::vector<angle::Radians>>);
  static_assert(fixed_pattern_collection<std::initializer_list<angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::tuple<Axis, int, Distance, angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::array<Dimensions<dynamic_size>, 5>>);
  static_assert(not fixed_pattern_collection<std::vector<int>>);
  static_assert(not fixed_pattern_collection<std::initializer_list<int>>);
}


#include "linear-algebra/coordinates/functions/compare_pattern_collections.hpp"

TEST(coordinates, compare_pattern_collections)
{
  static_assert(compare_pattern_collections(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}}));
  static_assert(compare_pattern_collections(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}, Axis{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}}));
  static_assert(compare_pattern_collections(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}, Axis{}}));
  static_assert(compare_pattern_collections<&stdcompat::is_lteq>(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}}));
  static_assert(compare_pattern_collections<&stdcompat::is_lt>(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>, Axis>{}, std::tuple{Polar{}, Axis{}}}));
  static_assert(compare_pattern_collections<&stdcompat::is_lt>(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Distance>{}, std::tuple<Dimensions<2>, Inclination<>>{}, std::tuple{Polar{}, Angle{}}}));
  static_assert(compare_pattern_collections<&stdcompat::is_gt>(
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Distance>{}, std::tuple<Dimensions<2>, Inclination<>>{}, std::tuple{Polar{}, Angle{}}},
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Axis>{}, Polar{}}));

  EXPECT_TRUE(compare_pattern_collections(
    std::tuple {std::tuple{Any{Distance{}}, Dimensions{3}}, std::tuple{Axis{}, Axis{}}, Polar{}},
    std::tuple {std::tuple{Distance{}, Axis{}, Axis{}, Axis{}}, std::tuple{Dimensions{2}}, Any{Polar{}}}));
  EXPECT_TRUE(compare_pattern_collections(
    std::tuple {std::vector{Any{Distance{}}, Any{Dimensions{3}}}, std::vector{Axis{}, Axis{}}, Polar{}},
    std::tuple {std::tuple{Distance{}, Axis{}, Axis{}, Axis{}}, std::tuple{Dimensions{2}}, std::vector{Polar{}}}));
  EXPECT_TRUE(compare_pattern_collections<&stdcompat::is_lt>(
    std::tuple {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}}, std::vector{Axis{}, Axis{}}, Polar{}},
    std::tuple {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}}));
  EXPECT_TRUE(compare_pattern_collections<&stdcompat::is_lt>(
    std::vector {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}}, std::vector{Any{Axis{}}, Any{Axis{}}}, std::vector{Any{Polar{}}}},
    std::vector {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}}));
}


#include "linear-algebra/coordinates/functions/internal/smallest_pattern.hpp"

TEST(coordinates, smallest_pattern)
{
  using OpenKalman::coordinates::internal::smallest_pattern;
  static_assert(smallest_pattern(std::tuple {Dimensions<3>{}, Dimensions<4>{}}) == 0);
  static_assert(smallest_pattern(std::tuple {Dimensions<3>{}, inclination::Radians{}}) == 1);
  static_assert(smallest_pattern(std::tuple {Dimensions<1>{}, angle::Radians{}}) == 0);
  static_assert(smallest_pattern(std::tuple {angle::Radians{}, Dimensions<1>{}, angle::Degrees{}}) == 0);
  static_assert(smallest_pattern(std::tuple {Polar{}, angle::Radians{}, angle::Degrees{}}) == 1);
  static_assert(smallest_pattern(std::tuple {Dimensions<3>{}, Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, angle::Degrees{}}) == 3);

  static_assert(smallest_pattern(std::tuple {Dimensions<3>{}, 4U}) == 0);
  static_assert(smallest_pattern(std::tuple {Dimensions<4>{}, 3U}) == 1);
  static_assert(smallest_pattern(std::tuple {4U, 3U, Dimensions<5>{}}) == 1);
  static_assert(smallest_pattern(std::tuple {4U, Dimensions{3}, 5U}) == 1);
  static_assert(smallest_pattern(std::tuple {4U, Spherical{}, Dimensions{3}, 5U}) == 1);
  EXPECT_TRUE(smallest_pattern(std::tuple {4U, Spherical{}, Any{Dimensions<2>{}}, 5U}) == 2);

  EXPECT_TRUE((smallest_pattern(std::vector{Any{Spherical{}}, Any{Dimensions<2>{}}, Any{Dimensions<4>{}}}) == 1));
  EXPECT_TRUE((smallest_pattern(std::vector{Any{Polar{}}, Any{Dimensions<3>{}}, Any{Dimensions<4>{}}}) == 0));
  EXPECT_TRUE(smallest_pattern(std::tuple {std::tuple {Dimensions<3>{}, Distance{}}, std::vector{Dimensions{2}, Dimensions{1}}, Dimensions<3>{}, std::tuple {angle::Radians{}, Distance{}}, std::tuple {angle::Degrees{}, Distance{}}}) == 3);
}


#include "linear-algebra/coordinates/functions/internal/largest_pattern.hpp"

TEST(coordinates, largest_pattern)
{
  using OpenKalman::coordinates::internal::largest_pattern;
  static_assert(largest_pattern(std::tuple {Dimensions<3>{}, Dimensions<4>{}}) == 1);
  static_assert(largest_pattern(std::tuple {Dimensions<3>{}, inclination::Radians{}}) == 0);
  static_assert(largest_pattern(std::tuple {Dimensions<1>{}, angle::Radians{}}) == 0);
  static_assert(largest_pattern(std::tuple {angle::Radians{}, Dimensions<1>{}, angle::Degrees{}}) == 0);
  static_assert(largest_pattern(std::tuple {angle::Radians{}, Polar{}, angle::Degrees{}}) == 1);
  static_assert(largest_pattern(std::tuple {Dimensions<3>{}, Dimensions<2>{}, Dimensions<4>{}, angle::Radians{}, angle::Degrees{}}) == 2);

  static_assert(largest_pattern(std::tuple {Dimensions<3>{}, 4U}) == 1);
  static_assert(largest_pattern(std::tuple {Dimensions<4>{}, 3U}) == 0);
  static_assert(largest_pattern(std::tuple {4U, 5U, Dimensions<3>{}}) == 1);
  static_assert(largest_pattern(std::tuple {4U, Dimensions{5}, 3U}) == 1);
  static_assert(largest_pattern(std::tuple {4U, Spherical{}, Dimensions{3}, 5U}) == 3);
  EXPECT_TRUE(largest_pattern(std::tuple {4U, Spherical{}, Any{Dimensions<6>{}}, 5U}) == 2);

  EXPECT_TRUE((largest_pattern(std::vector{Any{Spherical{}}, Any{Dimensions<5>{}}, Any{Dimensions<4>{}}}) == 1));
  EXPECT_TRUE((largest_pattern(std::vector{Any{Dimensions<1>{}}, Any{Polar{}}, Any{Dimensions<1>{}}}) == 1));
  EXPECT_TRUE(largest_pattern(std::tuple {std::tuple {Dimensions<3>{}, Distance{}}, std::vector{Dimensions{2}, Dimensions{3}}, Dimensions<3>{}, std::tuple {angle::Radians{}, Distance{}}, std::tuple {angle::Degrees{}, Distance{}}}) == 1);
}


#include "linear-algebra/coordinates/functions/internal/strip_1D_tail.hpp"

TEST(coordinates, strip_1D_tail)
{
  using OpenKalman::coordinates::internal::strip_1D_tail;
  auto d123 = std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}};
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{}), std::tuple{}));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{}}), std::tuple{}));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}}), d123));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}}), d123));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions<0>{}, Dimensions<1>{}}), std::tuple{Dimensions<1>{}, Dimensions<0>{}}));

  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Distance{}, Dimensions<1>{}, Dimensions<1>{}}), std::tuple{Distance{}}));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Distance{}, Dimensions<1>{}, Dimensions<1>{}}), std::tuple{Dimensions<1>{}, Distance{}}));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Distance{}, Angle{}, Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{}}), std::tuple{Distance{}, Angle{}}));

  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions{2}, Dimensions<3>{}, Dimensions<1>{}}), d123));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions{1}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}}), d123));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions{3}, Dimensions<1>{}}), std::tuple{Dimensions<1>{}, Dimensions{3}}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions{0}, Dimensions<1>{}}), std::tuple{Dimensions<1>{}, Dimensions<0>{}}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::vector{Any{Dimensions{1}}}), std::tuple{}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::vector{Any{Distance{}}, Any{Dimensions{1}}}), std::vector{Any{Distance{}}}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Dimensions{1}}}), std::tuple{Distance{}, Angle{}}));

  static_assert(compare_pattern_collections(strip_1D_tail(collections::views::repeat(Dimensions<1>{}, std::integral_constant<std::size_t, 10>{})), std::tuple{}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(stdcompat::ranges::views::repeat(Dimensions<1>{}, 10)), std::tuple{}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(stdcompat::ranges::views::repeat(Any{Dimensions<1>{}}, 10)), std::tuple{}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(stdcompat::ranges::views::repeat(Distance{}, 10)), stdcompat::ranges::views::repeat(Distance{}, 10)));
}


#include "linear-algebra/coordinates/functions/internal/most_fixed_pattern.hpp"

TEST(coordinates, most_fixed_pattern)
{
  using OpenKalman::coordinates::internal::most_fixed_pattern;
  static_assert(stdcompat::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Distance{}, Any{Distance{}}}))>, Distance>);
  static_assert(stdcompat::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Any{Distance{}}, Distance{}}))>, Distance>);
  static_assert(stdcompat::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Any{Distance{}}, Any{Distance{}}}))>, Any<>>);
  static_assert(stdcompat::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Any{Distance{}}, Distance{}, Any{Distance{}}}))>, Distance>);
}


#include "linear-algebra/coordinates/functions/internal/to_euclidean_pattern_collection.hpp"

TEST(coordinates, to_euclidean_pattern_collection)
{
  using OpenKalman::coordinates::internal::to_euclidean_pattern_collection;
  EXPECT_TRUE(compare_pattern_collections(to_euclidean_pattern_collection(std::tuple {4U, 2U, 5U}), std::tuple {4U, 2U, 5U}));
  EXPECT_TRUE(compare_pattern_collections(to_euclidean_pattern_collection(
    std::tuple {std::tuple<Distance, Dimensions<3>>{}, std::tuple<Axis, Angle<>>{}, Spherical{}}),
    std::tuple {4U, 2U, 3U}));
  EXPECT_TRUE(compare_pattern_collections(to_euclidean_pattern_collection(
    std::tuple {std::vector{Any{Distance{}}, Any{Dimensions{3}}}, std::vector{Axis{}, Axis{}}, Polar{}}),
    std::tuple {4U, 2U, 2U}));
  EXPECT_TRUE(compare_pattern_collections(to_euclidean_pattern_collection(
    std::vector {std::vector{Any{Distance{}}, Any{Dimensions{3}}}, std::vector{Any{Axis{}}, Any{Axis{}}}, std::vector{Any{Polar{}}}}),
    std::vector {4U, 2U, 2U}));
}

