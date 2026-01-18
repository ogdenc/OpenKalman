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
 * \brief Tests relating to \ref pattern_collection objects
 */

#include "collections/tests/tests.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Distance.hpp"
#include "patterns/descriptors/Angle.hpp"
#include "patterns/descriptors/Inclination.hpp"
#include "patterns/descriptors/Polar.hpp"
#include "patterns/descriptors/Spherical.hpp"
#include "patterns/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;

#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/concepts/fixed_pattern_collection.hpp"
#include "patterns/concepts/euclidean_pattern_collection.hpp"

TEST(patterns, pattern_collection)
{
  static_assert(collections::collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(collections::collection<std::vector<angle::Radians>>);

  static_assert(pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(pattern_collection<std::array<Distance, 5>>);
  static_assert(pattern_collection<std::vector<angle::Radians>>);
  static_assert(pattern_collection<std::initializer_list<angle::Radians>>);

  static_assert(not euclidean_pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(euclidean_pattern_collection<std::tuple<Axis, Dimensions<3>, unsigned, std::integral_constant<std::size_t, 5>>>);
  static_assert(not euclidean_pattern_collection<std::tuple<Axis, Dimensions<3>, unsigned, Distance, std::integral_constant<std::size_t, 5>>>);
  static_assert(euclidean_pattern_collection<std::array<Dimensions<4>, 5>>);
  static_assert(not euclidean_pattern_collection<std::array<Distance, 5>>);
  static_assert(euclidean_pattern_collection<std::vector<Axis>>);
  static_assert(not euclidean_pattern_collection<std::vector<angle::Radians>>);
  static_assert(euclidean_pattern_collection<std::initializer_list<unsigned>>);
  static_assert(not euclidean_pattern_collection<std::initializer_list<Distance>>);
  static_assert(euclidean_pattern_collection<std::tuple<Axis, Axis>[5]>);
  static_assert(not euclidean_pattern_collection<std::tuple<Axis, Distance>[5]>);

  static_assert(not fixed_pattern_collection<std::tuple<Axis, Distance, unsigned, angle::Radians>>);
  static_assert(fixed_pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::tuple<Axis, int, Distance, angle::Radians>>);
  static_assert(fixed_pattern_collection<std::array<Distance, 5>>);
  static_assert(not fixed_pattern_collection<std::array<Dimensions<stdex::dynamic_extent>, 5>>);
  static_assert(not fixed_pattern_collection<std::vector<angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::initializer_list<angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::vector<int>>);
  static_assert(not fixed_pattern_collection<std::initializer_list<int>>);
  static_assert(fixed_pattern_collection<angle::Radians[5]>);
  static_assert(not fixed_pattern_collection<unsigned[5]>);
  static_assert(not fixed_pattern_collection<Any<>[5]>);
}


#include "patterns/concepts/collection_compares_with.hpp"

TEST(patterns, collection_compares_with)
{
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Distance, Dimensions<3>>,    std::tuple<Axis, Axis>,    Polar<>>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis>, std::tuple<Dimensions<2>>, Polar<>, Axis>>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Distance, Dimensions<3>>,    std::tuple<Dimensions<>>,  Polar<>, Axis>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis>, std::tuple<Dimensions<2>>, Polar<>>>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Distance, Dimensions<3>>,    std::tuple<Dimensions<>>,  Polar<>, Dimensions<>>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis>, std::tuple<Dimensions<2>>, Polar<>>,
    &stdex::is_eq, applicability::permitted>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Distance, Dimensions<3>>,    std::tuple<Dimensions<>>,  Polar<>, Axis, Dimensions<>>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis>, std::tuple<Dimensions<2>>, Polar<>, Dimensions<>>,
    &stdex::is_eq, applicability::permitted>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>, Dimensions<>>>,
    std::tuple<std::tuple<Distance, Dimensions<3>>,              std::tuple<Axis, Axis>,                   Polar<>>,
    &stdex::is_gt>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>, Dimensions<>>, Dimensions<2>>,
    std::tuple<std::tuple<Distance, Dimensions<3>>,              std::tuple<Axis, Axis>,                   Polar<>>,
    &stdex::is_gt>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>, Dimensions<>>, Dimensions<0>>,
    std::tuple<std::tuple<Distance, Dimensions<3>>,              std::tuple<Axis, Axis>,                   Polar<>>,
    &stdex::is_gt, applicability::permitted>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>, Dimensions<>>>,
    std::tuple<std::tuple<Distance, Dimensions<3>>,              std::tuple<Axis, Axis>,                   Polar<>,                                    Dimensions<2>>,
    &stdex::is_gt, applicability::permitted>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>>,
    std::tuple<std::tuple<Distance, Dimensions<3>>,              std::tuple<Axis, Axis>,                   Polar<>,            Dimensions<>>,
    &stdex::is_gt>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>>,
    std::tuple<std::tuple<Any<>, Dimensions<3>>,                 std::tuple<Axis, Axis>,                   Polar<>>,
    &stdex::is_gt>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Any<>, Dimensions<3>>,                 std::tuple<Axis, Axis>,                   Polar<>, Dimensions<>, Dimensions<>>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>>,
    &stdex::is_lt>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Any<>, Dimensions<>>,                  std::tuple<Axis, Axis>,                   Polar<>>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>>,
    &stdex::is_lt, applicability::permitted>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Any<>, Dimensions<3>>,                 std::tuple<Axis, Axis>,                   Polar<>,                      Dimensions<2>>,
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>, Dimensions<3>, Dimensions<>>,
    &stdex::is_lt, applicability::permitted>);
  static_assert(collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>>,
    std::vector<std::vector<Any<>>>,
    &stdex::is_gt, applicability::permitted>);
  static_assert(not collection_compares_with<
    std::tuple<std::tuple<Distance, Axis, Axis, Axis, Distance>, std::tuple<Dimensions<2>, Inclination<>>, std::tuple<Polar<>, Angle<>>>,
    std::tuple<std::vector<Any<>>, Distance, std::vector<Any<>>>,
    &stdex::is_eq, applicability::permitted>);
}


#include "patterns/functions/compare_pattern_collections.hpp"

TEST(patterns, compare_pattern_collections)
{
  static_assert(compare_pattern_collections(
    std::tuple {std::tuple<Distance, Dimensions<3>>{},    std::tuple<Axis, Axis>{},    Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}}));
  static_assert(compare_pattern_collections(
    std::tuple {std::tuple<Distance, Dimensions<3>>{},    std::tuple<Axis, Axis>{},    Polar{}, Axis{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}}));
  static_assert(compare_pattern_collections(
    std::tuple {std::tuple<Distance, Dimensions<3>>{},    std::tuple<Axis, Axis>{},    Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}, Axis{}}));
  static_assert(compare_pattern_collections<&stdex::is_lteq>(
    std::tuple {std::tuple<Distance, Dimensions<3>>{},          std::tuple<Axis, Axis>{},    Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>>{}, Polar{}}));
  static_assert(compare_pattern_collections<&stdex::is_lt>(
    std::tuple {std::tuple<Distance, Dimensions<3>>{},          std::tuple<Axis, Axis>{},          Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Axis>{}, std::tuple<Dimensions<2>, Axis>{}, std::tuple{Polar{}, Axis{}}}));
  static_assert(compare_pattern_collections<&stdex::is_lt>(
    std::tuple {std::tuple<Distance, Dimensions<3>>{},              std::tuple<Axis, Axis>{},                   Polar{}},
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Distance>{}, std::tuple<Dimensions<2>, Inclination<>>{}, std::tuple{Polar{}, Angle{}}}));
  static_assert(compare_pattern_collections<&stdex::is_gt>(
    std::tuple {std::tuple<Distance, Axis, Axis, Axis, Distance>{}, std::tuple<Dimensions<2>, Inclination<>>{}, std::tuple{Polar{}, Angle{}}},
    std::tuple {std::tuple<Distance, Dimensions<3>>{},              std::tuple<Axis, Axis>{},                   Polar{}}));

  EXPECT_TRUE(compare_pattern_collections(
    std::tuple {std::tuple{Any{Distance{}}}, Dimensions{3},                  std::tuple{Axis{}, Axis{}},   Polar{}},
    std::tuple {std::tuple{Distance{}},      std::tuple{Axis{}, Axis{}, Axis{}}, std::tuple{Dimensions{2}}, Any{Polar{}}}));
  EXPECT_TRUE(compare_pattern_collections(
    std::tuple {std::vector{Any{Distance{}},                    Any{Dimensions{3}}},       std::vector{Axis{}, Axis{}},  Polar{}},
    std::tuple {std::tuple{Distance{}, Axis{}, Axis{}, Axis{}}, std::tuple{Dimensions{2}}, std::vector{Polar{}}}));
  EXPECT_TRUE(compare_pattern_collections<&stdex::is_lt>(
    std::tuple {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}},                                   std::vector{Axis{}, Axis{}},                           Polar{}},
    std::tuple {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}}));
  EXPECT_TRUE(compare_pattern_collections<&stdex::is_lt>(
    std::tuple {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}},                                   std::vector{Axis{}, Axis{}},                           Polar{}},
    std::tuple {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}, Dimensions<3>{}}));

  EXPECT_TRUE(compare_pattern_collections<&stdex::is_lt>(
    std::tuple  {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}},                                   std::vector{Any{Axis{}}, Any{Axis{}}},                 std::vector{Any{Polar{}}}},
    std::vector {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}, std::vector{Any{Dimensions<3>{}}}}));

  EXPECT_TRUE(compare_pattern_collections<&stdex::is_lt>(
    std::vector {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}},                                   std::vector{Any{Axis{}}, Any{Axis{}}},                 std::vector{Any{Polar{}}}},
    std::tuple  {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}, Dimensions<3>{}}));
  EXPECT_TRUE(compare_pattern_collections<&stdex::is_lt>(
    std::vector {std::vector{Any{Distance{}}, Any{Dimensions<3>{}}},                                   std::vector{Any{Axis{}}, Any{Axis{}}},                 std::vector{Any{Polar{}}}},
    std::vector {std::vector{Any{Distance{}}, Any{Axis{}}, Any{Axis{}}, Any{Axis{}}, Any{Distance{}}}, std::vector{Any{Dimensions<2>{}}, Any{Inclination{}}}, std::vector{Any{Polar{}}, Any{Angle{}}}, std::vector{Any{Dimensions<3>{}}}}));
}


#include "patterns/functions/internal/smallest_pattern.hpp"

TEST(patterns, smallest_pattern)
{
  using OpenKalman::patterns::internal::smallest_pattern;
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


#include "patterns/functions/internal/largest_pattern.hpp"

TEST(patterns, largest_pattern)
{
  using OpenKalman::patterns::internal::largest_pattern;
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


#include "patterns/functions/internal/most_fixed_pattern.hpp"

TEST(patterns, most_fixed_pattern)
{
  using OpenKalman::patterns::internal::most_fixed_pattern;
  static_assert(stdex::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Distance{}, Any{Distance{}}}))>, Distance>);
  static_assert(stdex::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Any{Distance{}}, Distance{}}))>, Distance>);
  static_assert(stdex::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Any{Distance{}}, Any{Distance{}}}))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(most_fixed_pattern(std::tuple{Any{Distance{}}, Distance{}, Any{Distance{}}}))>, Distance>);
}


#include "patterns/functions/internal/to_euclidean_pattern_collection.hpp"

TEST(patterns, to_euclidean_pattern_collection)
{
  using OpenKalman::patterns::internal::to_euclidean_pattern_collection;
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


#include "patterns/functions/get_pattern.hpp"
#include "patterns/traits/pattern_collection_element.hpp"

TEST(patterns, get_pattern)
{
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Distance{}, Angle{}, Polar{}}, std::integral_constant<std::size_t, 0>{}))>, Distance>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Distance{}, Angle{}, Polar{}}, std::integral_constant<std::size_t, 1>{}))>, Angle<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Distance{}, Angle{}, Polar{}}, std::integral_constant<std::size_t, 2>{}))>, Polar<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Distance{}, Angle{}, Polar{}}, std::integral_constant<std::size_t, 3>{}))>, Dimensions<1>>);

  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, std::integral_constant<std::size_t, 0>{}))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, std::integral_constant<std::size_t, 1>{}))>, Angle<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, std::integral_constant<std::size_t, 2>{}))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, std::integral_constant<std::size_t, 3>{}))>, Dimensions<1>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, 0U))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, 1U))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, 2U))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::tuple{Any{Distance{}}, Angle{}, Any{Polar{}}}, 3U))>, Any<>>);

  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, std::integral_constant<std::size_t, 0>{}))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, std::integral_constant<std::size_t, 1>{}))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, std::integral_constant<std::size_t, 2>{}))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, std::integral_constant<std::size_t, 3>{}))>, Any<>>);

  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, 0U))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, 1U))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, 2U))>, Any<>>);
  static_assert(stdex::same_as<std::decay_t<decltype(get_pattern(std::vector{Any{Distance{}}, Any{Angle{}}, Any{Polar{}}}, 3U))>, Any<>>);

  static_assert(stdex::same_as<pattern_collection_element_t<0, std::tuple<Distance, Angle<>, Polar<>>>, Distance>);
  static_assert(stdex::same_as<pattern_collection_element_t<1, std::tuple<Distance, Angle<>, Polar<>>>, Angle<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<2, std::tuple<Distance, Angle<>, Polar<>>>, Polar<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<3, std::tuple<Distance, Angle<>, Polar<>>>, Dimensions<1>>);

  static_assert(stdex::same_as<pattern_collection_element_t<0, std::tuple<Any<>, Angle<>, Any<>>>, Any<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<1, std::tuple<Any<>, Angle<>, Any<>>>, Angle<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<2, std::tuple<Any<>, Angle<>, Any<>>>, Any<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<3, std::tuple<Any<>, Angle<>, Any<>>>, Dimensions<1>>);

  static_assert(stdex::same_as<pattern_collection_element_t<0, std::vector<Any<>>>, Any<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<1, std::vector<Any<>>>, Any<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<2, std::vector<Any<>>>, Any<>>);
  static_assert(stdex::same_as<pattern_collection_element_t<3, std::vector<Any<>>>, Any<>>);
}


#include "patterns/concepts/collection_patterns_compare_with_dimension.hpp"

TEST(patterns, collection_patterns_compare_with_dimension)
{
  static_assert(collection_patterns_compare_with_dimension<std::tuple<>, 1>);

  using P0 = std::tuple<Distance, Dimensions<1>, Angle<>>;
  static_assert(collection_patterns_compare_with_dimension<P0, 1>);
  static_assert(collection_patterns_compare_with_dimension<P0, 1, &stdex::is_eq, 1>);
  static_assert(collection_patterns_compare_with_dimension<P0, 1, &stdex::is_eq, 2>);
  static_assert(collection_patterns_compare_with_dimension<P0, 1, &stdex::is_eq, 3>);
  static_assert(collection_patterns_compare_with_dimension<P0, 1, &stdex::is_eq, 4>);
  static_assert(collection_patterns_compare_with_dimension<P0, 2, &stdex::is_lt>);

  using P1 = std::tuple<Dimensions<2>, Polar<>, Spherical<>>;
  static_assert(collection_patterns_compare_with_dimension<P1, 2, &stdex::is_gteq>);
  static_assert(collection_patterns_compare_with_dimension<P1, 3, &stdex::is_lteq>);

  using P2 = std::tuple<Distance, Dimensions<>, Angle<>>;
  static_assert(not collection_patterns_compare_with_dimension<P2, 1>);
  static_assert(collection_patterns_compare_with_dimension<P2, 1, &stdex::is_eq, 1>);
  static_assert(not collection_patterns_compare_with_dimension<P2, 1, &stdex::is_eq, 2>);
  static_assert(collection_patterns_compare_with_dimension<P2, 1, &stdex::is_eq, values::unbounded_size, applicability::permitted>);
  static_assert(not collection_patterns_compare_with_dimension<P2, 2, &stdex::is_lt>);
  static_assert(collection_patterns_compare_with_dimension<P2, 2, &stdex::is_lt, values::unbounded_size, applicability::permitted>);

  using P3 = std::vector<Any<>>;
  static_assert(not collection_patterns_compare_with_dimension<P3, 1>);
  static_assert(collection_patterns_compare_with_dimension<P3, 1, &stdex::is_eq, values::unbounded_size, applicability::permitted>);
  static_assert(not collection_patterns_compare_with_dimension<P3, 2, &stdex::is_lt>);
  static_assert(collection_patterns_compare_with_dimension<P3, 2, &stdex::is_lt, values::unbounded_size, applicability::permitted>);

  using P4 = stdex::ranges::repeat_view<Distance>;
  static_assert(collection_patterns_compare_with_dimension<P4, 1>);
  static_assert(collection_patterns_compare_with_dimension<P4, 2, &stdex::is_lt>);

  using P5 = stdex::ranges::repeat_view<Dimensions<>>;
  static_assert(not collection_patterns_compare_with_dimension<P5, 1>);
  static_assert(collection_patterns_compare_with_dimension<P5, 1, &stdex::is_eq, values::unbounded_size, applicability::permitted>);
  static_assert(not collection_patterns_compare_with_dimension<P5, 2, &stdex::is_lt>);
  static_assert(collection_patterns_compare_with_dimension<P5, 2, &stdex::is_lt, values::unbounded_size, applicability::permitted>);

  using P6 = std::tuple<Dimensions<0>, Dimensions<0>>;
  static_assert(collection_patterns_compare_with_dimension<P6, 0>);
  static_assert(collection_patterns_compare_with_dimension<P6, 1, &stdex::is_lt>);
}


#include "patterns/functions/compare_collection_patterns_with_dimension.hpp"

TEST(patterns, compare_collection_patterns_with_dimension)
{
  static_assert(compare_collection_patterns_with_dimension<1>(std::tuple{}));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<1>(std::vector<Polar<>>{})));

  auto p1 = std::tuple{Distance{}, Dimensions<1>{}, Angle{}};
  static_assert(compare_collection_patterns_with_dimension<1, &stdex::is_eq, 1>(p1));
  static_assert(compare_collection_patterns_with_dimension<1, &stdex::is_eq, 2>(p1));
  static_assert(compare_collection_patterns_with_dimension<1, &stdex::is_eq, 3>(p1));
  static_assert(compare_collection_patterns_with_dimension<1, &stdex::is_eq, 4>(p1));
  static_assert(compare_collection_patterns_with_dimension<1>(p1));
  static_assert(compare_collection_patterns_with_dimension<2, &stdex::is_lt>(p1));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_eq>(p1, 1U)));

  auto p2 = std::tuple{Dimensions<2>{}, Polar{}, Spherical{}};
  static_assert(compare_collection_patterns_with_dimension<2, &stdex::is_eq, 1>(p2));
  static_assert(compare_collection_patterns_with_dimension<2, &stdex::is_eq, 2>(p2));
  static_assert(not compare_collection_patterns_with_dimension<2, &stdex::is_eq, 3>(p2));
  static_assert(not compare_collection_patterns_with_dimension<2, &stdex::is_eq, 4>(p2));
  static_assert(not compare_collection_patterns_with_dimension<2>(p2));
  static_assert(compare_collection_patterns_with_dimension<3, &stdex::is_lteq>(p2));
  static_assert(compare_collection_patterns_with_dimension<4, &stdex::is_lteq>(p2));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_eq, 1>(p2, 2U)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_eq, 2>(p2, 2U)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<&stdex::is_eq, 3>(p2, 2U)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_lteq, 3>(p2, 3U)));

  auto p3 = std::vector{Any{Polar{}}, Any{Dimensions<2>{}}, Any{Spherical{}}};
  EXPECT_TRUE((compare_collection_patterns_with_dimension<2, &stdex::is_eq, 1>(p3)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<2, &stdex::is_eq, 2>(p3)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<2, &stdex::is_eq, 3>(p3)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<2, &stdex::is_eq, 4>(p3)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<2>(p3)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<3, &stdex::is_lteq>(p3)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<4, &stdex::is_lteq>(p3)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension(p3, 2U)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_lteq>(p3, 3U)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_lteq>(p3, 4U)));

  auto p4 = stdex::ranges::views::repeat(Polar{});
  static_assert(compare_collection_patterns_with_dimension<2, &stdex::is_eq, 1>(p4));
  static_assert(compare_collection_patterns_with_dimension<2, &stdex::is_eq, 100>(p4));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_eq, 1>(p4, 2U)));
  EXPECT_TRUE((compare_collection_patterns_with_dimension<&stdex::is_eq, 100>(p4, 2U)));

  auto p5 = stdex::ranges::views::repeat(Any{Spherical{}});
  EXPECT_FALSE((compare_collection_patterns_with_dimension<3, &stdex::is_eq, 1>(p5)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<3, &stdex::is_eq, 100>(p5)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<&stdex::is_eq, 1>(p5, 3U)));
  EXPECT_FALSE((compare_collection_patterns_with_dimension<&stdex::is_eq, 100>(p5, 3U)));
}


#include "patterns/concepts/collection_patterns_have_same_dimension.hpp"

TEST(patterns, collection_patterns_have_same_dimension)
{
  static_assert(collection_patterns_have_same_dimension<std::tuple<>>);

  using P1 = std::tuple<Distance, Dimensions<1>, Angle<>>;
  static_assert(collection_patterns_have_same_dimension<P1, 1>);
  static_assert(collection_patterns_have_same_dimension<P1, 2>);
  static_assert(collection_patterns_have_same_dimension<P1, 3>);
  static_assert(collection_patterns_have_same_dimension<P1, 4>);
  static_assert(collection_patterns_have_same_dimension<P1>);

  using P2 = std::tuple<Dimensions<0>, Dimensions<0>>;
  static_assert(collection_patterns_have_same_dimension<P2, 1>);
  static_assert(collection_patterns_have_same_dimension<P2, 2>);
  static_assert(not collection_patterns_have_same_dimension<P2, 3, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P2>);

  using P3 = std::tuple<Distance, Dimensions<>, Angle<>>;
  static_assert(collection_patterns_have_same_dimension<P3, 1>);
  static_assert(not collection_patterns_have_same_dimension<P3, 2>);
  static_assert(collection_patterns_have_same_dimension<P3, 2, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P3, 3, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P3, 4, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P3, values::unbounded_size, applicability::permitted>);

  using P4 = std::tuple<Spherical<>, Dimensions<3>, Dimensions<>, std::tuple<Distance, Spherical<>>>;
  static_assert(collection_patterns_have_same_dimension<P4, 1>);
  static_assert(collection_patterns_have_same_dimension<P4, 2>);
  static_assert(not collection_patterns_have_same_dimension<P4, 3>);
  static_assert(collection_patterns_have_same_dimension<P4, 3, applicability::permitted>);
  static_assert(not collection_patterns_have_same_dimension<P4, 4, applicability::permitted>);
  static_assert(not collection_patterns_have_same_dimension<P4, 5, applicability::permitted>);
  static_assert(not collection_patterns_have_same_dimension<P4, values::unbounded_size, applicability::permitted>);

  using P5 = std::vector<Distance>;
  static_assert(collection_patterns_have_same_dimension<P5, 1_uz>);
  static_assert(collection_patterns_have_same_dimension<P5, 2>);
  static_assert(collection_patterns_have_same_dimension<P5, 100>);
  static_assert(collection_patterns_have_same_dimension<P5>);

  using P6 = std::vector<Any<>>;
  static_assert(collection_patterns_have_same_dimension<P6, 1>);
  static_assert(collection_patterns_have_same_dimension<P6, 1, applicability::permitted>);
  static_assert(not collection_patterns_have_same_dimension<P6, 2>);
  static_assert(collection_patterns_have_same_dimension<P6, 2, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P6, 100, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P6, values::unbounded_size, applicability::permitted>);

  using P7 = stdex::ranges::repeat_view<Polar<>>;
  static_assert(collection_patterns_have_same_dimension<P7, 1>);
  static_assert(collection_patterns_have_same_dimension<P7, 2>);
  static_assert(collection_patterns_have_same_dimension<P7, 100>);
  static_assert(collection_patterns_have_same_dimension<P7>);

  using P8 = stdex::ranges::repeat_view<Any<>>;
  static_assert(collection_patterns_have_same_dimension<P8, 1>);
  static_assert(collection_patterns_have_same_dimension<P8, 1, applicability::permitted>);
  static_assert(not collection_patterns_have_same_dimension<P8, 2>);
  static_assert(collection_patterns_have_same_dimension<P8, 2, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P8, 100, applicability::permitted>);
  static_assert(collection_patterns_have_same_dimension<P8, values::unbounded_size, applicability::permitted>);
}


#include "patterns/functions/get_common_pattern_collection_dimension.hpp"

TEST(patterns, get_common_pattern_collection_dimension)
{
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension(std::tuple{}))::value_type> == 1);
  EXPECT_EQ(*get_common_pattern_collection_dimension(std::vector<Polar<>>{}), 1);

  auto p1 = std::tuple{Distance{}, Dimensions<1>{}, Angle{}};
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<1>(p1))::value_type> == 1);
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<2>(p1))::value_type> == 1);
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<3>(p1))::value_type> == 1);
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<4>(p1))::value_type> == 1);
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension(p1))::value_type> == 1);
  static_assert(get_common_pattern_collection_dimension(p1));

  auto p2 = std::tuple{Dimensions<0>{}, Dimensions<0>{}};
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<1>(p2))::value_type> == 0);
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<2>(p2))::value_type> == 0);
  static_assert(not get_common_pattern_collection_dimension<3>(p2));
  static_assert(get_common_pattern_collection_dimension(p2));

  constexpr auto p3 = std::tuple{Distance{}, Dimensions{1}, Angle{}};
  static_assert(values::fixed_value_of_v<decltype(get_common_pattern_collection_dimension<1>(p3))::value_type> == 1);
  static_assert(*get_common_pattern_collection_dimension<2>(p3) == 1);
  static_assert(*get_common_pattern_collection_dimension<3>(p3) == 1);
  static_assert(*get_common_pattern_collection_dimension<4>(p3) == 1);
  static_assert(*get_common_pattern_collection_dimension(p3) == 1);
  static_assert(get_common_pattern_collection_dimension(p3));

  auto p4 = std::vector{Distance{}, Distance{}, Distance{}};
  EXPECT_EQ(*get_common_pattern_collection_dimension<1>(p4), 1);
  EXPECT_EQ(*get_common_pattern_collection_dimension<3>(p4), 1);
  EXPECT_EQ(*get_common_pattern_collection_dimension<100>(p4), 1);
  EXPECT_EQ(*get_common_pattern_collection_dimension(p4), 1);

  auto p5 = std::vector{Polar{}, Polar{}, Polar{}};
  EXPECT_EQ(*get_common_pattern_collection_dimension<1>(p5), 2);
  EXPECT_EQ(*get_common_pattern_collection_dimension<3>(p5), 2);
  EXPECT_EQ(*get_common_pattern_collection_dimension(p5), 2);
  EXPECT_FALSE(get_common_pattern_collection_dimension<4>(p5));

  auto p6 = std::vector{Any{Polar{}}, Any{Dimensions<2>{}}, Any{Polar{}}};
  EXPECT_EQ(*get_common_pattern_collection_dimension<1>(p6), 2);
  EXPECT_EQ(*get_common_pattern_collection_dimension<3>(p6), 2);
  EXPECT_EQ(*get_common_pattern_collection_dimension(p6), 2);
  EXPECT_FALSE(get_common_pattern_collection_dimension<4>(p6));

  auto p7 = std::vector{Any{Distance{}}, Any{Dimensions<2>{}}, Any{Angle{}}};
  EXPECT_EQ(*get_common_pattern_collection_dimension<1>(p7), 1);
  EXPECT_FALSE(get_common_pattern_collection_dimension<2>(p7));
  EXPECT_FALSE(get_common_pattern_collection_dimension<3>(p7));
  EXPECT_FALSE(get_common_pattern_collection_dimension<4>(p7));
  EXPECT_FALSE(get_common_pattern_collection_dimension(p7));

  auto p8 = stdex::ranges::views::repeat(Polar{});
  static_assert(*get_common_pattern_collection_dimension<1>(p8) == 2);
  static_assert(*get_common_pattern_collection_dimension<3>(p8) == 2);
  static_assert(*get_common_pattern_collection_dimension<100>(p8) == 2);
  static_assert(*get_common_pattern_collection_dimension(p8) == 2);

  auto p9 = stdex::ranges::views::repeat(Any{Spherical{}});
  EXPECT_FALSE(get_common_pattern_collection_dimension<1>(p9));
  EXPECT_FALSE(get_common_pattern_collection_dimension<100>(p9));

}


#include "patterns/functions/to_extents.hpp"

TEST(patterns, extents)
{
  static_assert(pattern_collection<stdex::extents<std::size_t, 2, 3, 4>>);
  static_assert(euclidean_pattern_collection<stdex::extents<std::size_t, 2, 3, 4>>);
  static_assert(fixed_pattern_collection<stdex::extents<std::size_t, 2, 3, 4>>);

  static_assert(pattern_collection<stdex::extents<std::size_t, 2, stdex::dynamic_extent, 4>>);
  static_assert(euclidean_pattern_collection<stdex::extents<std::size_t, stdex::dynamic_extent, 3, 4>>);
  static_assert(not fixed_pattern_collection<stdex::extents<std::size_t, 2, 3, stdex::dynamic_extent>>);

  static_assert(collections::get<0>(stdex::extents<std::size_t, 2, 3, 4>{}) == 2);
  static_assert(collections::get<1>(stdex::extents<std::size_t, 2, stdex::dynamic_extent, 4>{3}) == 3);

  static_assert(stdex::same_as<decltype(to_extents(std::declval<stdex::extents<std::size_t>>())), stdex::extents<std::size_t>>);
  static_assert(stdex::same_as<decltype(to_extents<1>(std::declval<stdex::extents<std::size_t>>())), stdex::extents<std::size_t, 1>>);
  static_assert(stdex::same_as<decltype(to_extents<2>(std::declval<stdex::extents<std::size_t>>())), stdex::extents<std::size_t, 1, 1>>);
  static_assert(stdex::same_as<decltype(to_extents(std::declval<std::tuple<>>())), stdex::extents<std::size_t>>);
  static_assert(stdex::same_as<decltype(to_extents<3>(std::declval<std::tuple<>>())), stdex::extents<std::size_t, 1, 1, 1>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<stdex::extents<std::size_t, 1, 1, 1>>())),
    stdex::extents<std::size_t>>);
  static_assert(stdex::same_as<
    decltype(to_extents<3>(std::declval<stdex::extents<std::size_t, 1, 1, 1>>())),
    stdex::extents<std::size_t, 1, 1, 1>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<std::tuple<Dimensions<1>, Dimensions<1>, Dimensions<1>>>())),
    stdex::extents<std::size_t>>);
  static_assert(stdex::same_as<
    decltype(to_extents<4>(std::declval<std::tuple<Dimensions<1>, Dimensions<1>, Dimensions<1>>>())),
    stdex::extents<std::size_t, 1, 1, 1, 1>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<stdex::extents<std::size_t, 2, 3, 1, 1>>())),
    stdex::extents<std::size_t, 2, 3>>);
  static_assert(stdex::same_as<
    decltype(to_extents<3>(std::declval<stdex::extents<std::size_t, 2, 3, 1, 1>>())),
    stdex::extents<std::size_t, 2, 3, 1>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<std::tuple<Dimensions<2>, Dimensions<3>, Dimensions<1>, Dimensions<1>>>())),
    stdex::extents<std::size_t, 2, 3>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<stdex::extents<std::size_t, 4, 2, 3>>())),
    stdex::extents<std::size_t, 4, 2, 3>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<std::tuple<Dimensions<4>, Dimensions<2>, Dimensions<3>>>())),
    stdex::extents<std::size_t, 4, 2, 3>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<std::tuple<std::tuple<Distance, Dimensions<3>>, std::tuple<Axis, Axis>, Polar<>>>())),
    stdex::extents<std::size_t, 4, 2, 2>>);
  static_assert(stdex::same_as<
    decltype(to_extents(std::declval<std::tuple<Dimensions<>, std::tuple<Axis, Axis>, Spherical<>>>())),
    stdex::extents<std::size_t, stdex::dynamic_extent, 2, 3>>);
  static_assert(collections::get<0>(to_extents(std::tuple{4U, 5U, 6U})) == 4);
  static_assert(collections::get<2>(to_extents(std::tuple{4U, 5U, Dimensions<6>{}})) == 6);
  static_assert(collections::size_of_v<decltype(to_extents(stdex::extents<std::size_t, 4, 5, 6, 1>{}))> == 3);
  static_assert(collections::size_of_v<decltype(to_extents(stdex::extents<std::size_t, 4, 5, 6, stdex::dynamic_extent>{}))> == 4);
  static_assert(collections::size_of_v<decltype(to_extents(std::tuple{4U, 5U, 6U, Dimensions<1>{}}))> == 3);
  static_assert(collections::size_of_v<decltype(to_extents(std::tuple{4U, 5U, 6U, Dimensions{1}}))> == 4);
}


#include "patterns/views/diagonal_of.hpp"

TEST(patterns, diagonal_of)
{
  using patterns::views::diagonal_of;

  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{Dimensions<3>{}, Dimensions<3>{}, Angle{}, Polar{}})),
    std::tuple<Dimensions<3>, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{Dimensions<2>{}, Dimensions<3>{}, Angle{}, Polar{}})),
    std::tuple<Dimensions<2>, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{Dimensions<3>{}, Dimensions<2>{}, Angle{}, Polar{}})),
    std::tuple<Dimensions<2>, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{Dimensions<0>{}, Dimensions<3>{}, Angle{}, Polar{}})),
    std::tuple<Dimensions<0>, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{Dimensions<3>{}, Dimensions<0>{}, Angle{}, Polar{}})),
    std::tuple<Dimensions<0>, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{Dimensions<3>{}})),
    std::tuple<Dimensions<1>>>);
  static_assert(collection_compares_with<
    decltype(diagonal_of(std::tuple{})),
    std::tuple<>>);

  EXPECT_TRUE(compare_pattern_collections(diagonal_of(
    std::tuple {Dimensions{3}, Dimensions{3}, Angle{}, Polar{}}),
    std::tuple {Dimensions{3}, Angle{}, Polar{}}));
  EXPECT_TRUE(compare_pattern_collections(diagonal_of(
    std::tuple {Dimensions{2}, Dimensions{3}, Angle{}, Polar{}}),
    std::tuple {Dimensions{2}, Angle{}, Polar{}}));
  EXPECT_TRUE(compare_pattern_collections(diagonal_of(
    std::tuple {Dimensions{3}, Dimensions{2}, Angle{}, Polar{}}),
    std::tuple {Dimensions{2}, Angle{}, Polar{}}));
  EXPECT_TRUE(compare_pattern_collections(diagonal_of(
    std::tuple {Dimensions{0}, Dimensions{3}, Angle{}, Polar{}}),
    std::tuple {Dimensions{0}, Angle{}, Polar{}}));
  EXPECT_TRUE(compare_pattern_collections(diagonal_of(
    std::tuple {Dimensions{3}, Dimensions{0}, Angle{}, Polar{}}),
    std::tuple {Dimensions{0}, Angle{}, Polar{}}));
  EXPECT_TRUE(compare_pattern_collections(diagonal_of(
    std::tuple {Dimensions{3}}),
    std::tuple {Dimensions{1}}));

  EXPECT_TRUE(compare_pattern_collections(
    diagonal_of(std::tuple {Dimensions<2>{}, std::tuple{Dimensions<2>{}, Distance{}}}),
    std::tuple{Dimensions<2>{}}));
  EXPECT_TRUE(compare_pattern_collections(
    diagonal_of(std::tuple {std::tuple{Dimensions<2>{}, Distance{}}, Dimensions<2>{}}),
    std::tuple{Dimensions<2>{}}));

  EXPECT_TRUE(compare_pattern_collections(
    diagonal_of(std::tuple {Polar{}, std::tuple{Polar{}, Distance{}}}),
    std::tuple{Polar{}}));
  EXPECT_TRUE(compare_pattern_collections(
    diagonal_of(std::tuple {std::tuple{Polar{}, Distance{}}, Polar{}}),
    std::tuple{Polar{}}));

}


#include "patterns/views/to_diagonal.hpp"

TEST(patterns, to_diagonal)
{
  using patterns::views::to_diagonal;

  static_assert(collection_compares_with<
    decltype(to_diagonal(std::tuple{Dimensions<3>{}, Angle{}, Polar{}})),
    std::tuple<Dimensions<3>, Dimensions<3>, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(to_diagonal(std::tuple{Distance{}, Angle{}, Polar{}})),
    std::tuple<Distance, Distance, Angle<>, Polar<>>>);
  static_assert(collection_compares_with<
    decltype(to_diagonal(std::tuple{})),
    std::tuple<>>);
}


#include "patterns/views/transpose.hpp"

TEST(patterns, transpose)
{
  using patterns::views::transpose;

  static_assert(collection_compares_with<decltype(transpose<>(std::tuple{})), std::tuple<>>);

  using TA = decltype(transpose<>(std::tuple{Angle{}}));
  static_assert(collections::size_of_v<TA> == 2);
  static_assert(collection_compares_with<TA, std::tuple<Dimensions<1>, Angle<>>>);
  EXPECT_TRUE(compare_pattern_collections(transpose<>(std::tuple {Dimensions{3}}), std::tuple {Dimensions<1>{}, Dimensions{3}}));

  using TB = decltype(transpose<>(std::tuple{Dimensions<3>{}, Angle{}, Polar{}}));
  static_assert(collection_compares_with<TB, std::tuple<Angle<>, Dimensions<3>, Polar<>>>);
  EXPECT_TRUE(compare_pattern_collections(transpose<>(std::tuple {Dimensions{3}, Angle{}, Polar{}}), std::tuple {Angle{}, Dimensions{3}, Polar{}}));

  using TC = decltype(transpose<>(std::tuple{Dimensions<3>{}, Angle{}}));
  static_assert(collection_compares_with<TC, std::tuple<Angle<>, Dimensions<3>>>);
  EXPECT_TRUE(compare_pattern_collections(transpose<>(std::tuple {Dimensions{3}, Angle{}}), std::tuple {Angle{}, Dimensions{3}}));

  using TD = decltype(transpose<0, 2>(std::tuple{Dimensions<3>{}, Angle{}, Polar{}, Distance{}}));
  static_assert(collection_compares_with<TD, std::tuple<Polar<>, Angle<>, Dimensions<3>, Distance>>);

  using TE = decltype(transpose<1, 2>(std::tuple{Dimensions<3>{}, Angle{}, Polar{}, Distance{}}));
  static_assert(collection_compares_with<TE, std::tuple<Dimensions<3>, Polar<>, Angle<>, Distance>>);

  using TF = decltype(transpose<1, 3>(std::tuple{Dimensions<3>{}, Angle{}, Polar{}, Distance{}, Spherical{}}));
  static_assert(collection_compares_with<TF, std::tuple<Dimensions<3>, Distance, Polar<>, Angle<>, Spherical<>>>);

  using TG = decltype(transpose<1, 2>(std::tuple{Dimensions<3>{}, Angle{}}));
  static_assert(collections::size_of_v<TG> == 3);
  static_assert(collection_compares_with<TG, std::tuple<Dimensions<3>, Dimensions<1>, Angle<>>>);

  using TH = decltype(transpose<2, 4>(std::tuple{Dimensions<3>{}, Angle{}}));
  static_assert(collections::size_of_v<TH> == 2);
  static_assert(collection_compares_with<TH, std::tuple<Dimensions<3>, Angle<>>>);

  auto z0 = transpose<>(std::vector<Any<>> {});
  EXPECT_EQ(collections::get_size(z0), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<0>(z0)), 1);
  EXPECT_EQ(patterns::get_dimension(get_pattern<1>(z0)), 1);
  EXPECT_TRUE(compare(get_pattern<0>(z0), Dimensions<1>{}));
  EXPECT_TRUE(compare(get_pattern<1>(z0), Dimensions<1>{}));
  EXPECT_TRUE(compare_pattern_collections(std::move(z0), std::tuple {}));

  auto z1 = transpose<>(std::vector {Any{Polar{}}});
  EXPECT_EQ(collections::get_size(z1), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<0>(z1)), 1);
  EXPECT_EQ(patterns::get_dimension(get_pattern<1>(z1)), 2);
  EXPECT_TRUE(compare(get_pattern<0>(z1), Dimensions<1>{}));
  EXPECT_TRUE(compare(get_pattern<1>(z1), Polar{}));
  EXPECT_TRUE(compare_pattern_collections(std::move(z1), std::tuple {Dimensions<1>{}, Polar{}}));

  auto z2 = transpose<>(std::vector {Any{Dimensions{3}}, Any{Polar{}}});
  EXPECT_EQ(collections::get_size(z2), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<0>(z2)), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<1>(z2)), 3);
  EXPECT_TRUE(compare(get_pattern<0>(z2), Polar{}));
  EXPECT_TRUE(compare(get_pattern<1>(z2), Dimensions{3}));
  EXPECT_TRUE(compare_pattern_collections(std::move(z2), std::tuple {Polar{}, Dimensions{3}}));

  auto z3 = transpose<>(std::vector {Any{Dimensions{3}}, Any{Polar{}}, Any{Angle{}}});
  EXPECT_EQ(collections::get_size(z3), 3);
  EXPECT_EQ(patterns::get_dimension(get_pattern<0>(z3)), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<1>(z3)), 3);
  EXPECT_EQ(patterns::get_dimension(get_pattern<2>(z3)), 1);
  EXPECT_TRUE(compare(get_pattern<0>(z3), Polar{}));
  EXPECT_TRUE(compare(get_pattern<1>(z3), Dimensions{3}));
  EXPECT_TRUE(compare(get_pattern<2>(z3), Angle{}));
  EXPECT_TRUE(compare_pattern_collections(std::move(z3), std::tuple {Polar{}, Dimensions{3}, Angle{}}));

  auto z4 = transpose<0, 2>(std::vector {Any{Dimensions{3}}, Any{Polar{}}, Any{Angle{}}});
  EXPECT_EQ(collections::get_size(z4), 3);
  EXPECT_EQ(patterns::get_dimension(get_pattern<0>(z4)), 1);
  EXPECT_EQ(patterns::get_dimension(get_pattern<1>(z4)), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<2>(z4)), 3);
  EXPECT_EQ(patterns::get_dimension(get_pattern<3>(z4)), 1);
  EXPECT_TRUE(compare(get_pattern<0>(z4), Angle{}));
  EXPECT_TRUE(compare(get_pattern<1>(z4), Polar{}));
  EXPECT_TRUE(compare(get_pattern<2>(z4), Dimensions{3}));
  EXPECT_TRUE(compare(get_pattern<3>(z4), Dimensions<1>{}));
  EXPECT_TRUE(compare_pattern_collections(std::move(z4), std::tuple {Angle{}, Polar{}, Dimensions{3}}));

  auto z5 = transpose<1, 3>(std::vector {Any{Distance{}}, Any{Dimensions{3}}, Any{Polar{}}, Any{Angle{}}});
  EXPECT_EQ(collections::get_size(z5), 4);
  EXPECT_EQ(patterns::get_dimension(get_pattern<0>(z5)), 1);
  EXPECT_EQ(patterns::get_dimension(get_pattern<1>(z5)), 1);
  EXPECT_EQ(patterns::get_dimension(get_pattern<2>(z5)), 2);
  EXPECT_EQ(patterns::get_dimension(get_pattern<3>(z5)), 3);
  EXPECT_EQ(patterns::get_dimension(get_pattern<4>(z5)), 1);
  EXPECT_TRUE(compare(get_pattern<0>(z5), Distance{}));
  EXPECT_TRUE(compare(get_pattern<1>(z5), Angle{}));
  EXPECT_TRUE(compare(get_pattern<2>(z5), Polar{}));
  EXPECT_TRUE(compare(get_pattern<3>(z5), Dimensions{3}));
  EXPECT_TRUE(compare(get_pattern<4>(z5), Dimensions<1>{}));
  EXPECT_TRUE(compare_pattern_collections(std::move(z5), std::tuple {Distance{}, Angle{}, Polar{}, Dimensions{3}}));
}


#include "patterns/functions/internal/strip_1D_tail.hpp"

TEST(patterns, strip_1D_tail)
{
  using OpenKalman::patterns::internal::strip_1D_tail;
  auto d123 = std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}};
  static_assert(compare_pattern_collections(strip_1D_tail(stdex::extents<std::size_t>{}), std::tuple{}));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{}), std::tuple{}));
  static_assert(compare_pattern_collections(strip_1D_tail(stdex::extents<std::size_t, 1, 1, 1>{}), std::tuple{}));
  static_assert(compare_pattern_collections(strip_1D_tail(std::tuple{Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{}}), std::tuple{}));
  static_assert(compare_pattern_collections(strip_1D_tail(stdex::extents<std::size_t, 1, 2, 3, 1, 1>{}), d123));
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
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(stdex::ranges::views::repeat(Dimensions<1>{}, 10)), std::tuple{}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(stdex::ranges::views::repeat(Any{Dimensions<1>{}}, 10)), std::tuple{}));
  EXPECT_TRUE(compare_pattern_collections(strip_1D_tail(stdex::ranges::views::repeat(Distance{}, 10)), stdex::ranges::views::repeat(Distance{}, 10)));
}

