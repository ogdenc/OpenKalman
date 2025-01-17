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
 * \brief Tests for descriptor::static_concatenate
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"
#include "linear-algebra/vector-space-descriptors/functions/arithmetic-operators.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, fixed_concatenation)
{
  static_assert(Dimensions<3>{} + Dimensions<4>{} == Dimensions<7>{});
  static_assert(StaticDescriptor<Axis, Axis>{} + StaticDescriptor<Axis, Axis, Axis>{} == Dimensions<5>{});
  static_assert(Polar<Distance, angle::Radians>{} + Dimensions<2>{} == StaticDescriptor<Polar<Distance, angle::Radians>, Dimensions<2>>{});

  static_assert(StaticDescriptor<>{} + StaticDescriptor<>{} == StaticDescriptor<>{});
  static_assert(StaticDescriptor<Axis>{} + StaticDescriptor<>{} == Axis{});
  static_assert(StaticDescriptor<>{} + StaticDescriptor<Axis>{} == Axis{});
  static_assert(Axis{} + StaticDescriptor<angle::Radians>{} == StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis>{} + StaticDescriptor<angle::Radians>{} == StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} + StaticDescriptor<angle::Radians, Axis>{} == StaticDescriptor<Axis, angle::Radians, angle::Radians, Axis>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} + StaticDescriptor<angle::Radians, Axis>{} + StaticDescriptor<Axis, angle::Radians>{} ==
    StaticDescriptor<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis>{} + Polar<>{} == StaticDescriptor<Axis, Polar<>>{});
  static_assert(Polar<>{} + StaticDescriptor<Axis>{} == StaticDescriptor<Polar<>, Axis>{});
  static_assert(Polar<>{} + Spherical<>{} + Polar<>{} == StaticDescriptor<Polar<>, Spherical<>, Polar<>>{});

  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Dimensions<0>, Distance, Dimensions<2>>{} + StaticDescriptor<Axis, Dimensions<2>, inclination::Radians>{})),
    std::tuple<Distance, Dimensions<5>, inclination::Radians>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Dimensions<0>, Distance, Dimensions<0>>{} + StaticDescriptor<Dimensions<0>, inclination::Radians>{})),
    std::tuple<Distance, inclination::Radians>>);
  static_assert(std::is_same_v<decltype(get_collection_of(Dimensions<0>{} + StaticDescriptor<inclination::Radians>{})), std::tuple<inclination::Radians>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<inclination::Radians>{} + Dimensions<0>{})), std::tuple<inclination::Radians>>);
}


TEST(descriptors, dynamic_concatenation)
{
  static_assert(Dimensions{3} + Dimensions{4} == Dimensions{7});

  auto ca = get_collection_of(DynamicDescriptor<double> {Axis{}, angle::Radians{}} + DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}, Dimensions<2>{}});
  auto ita = std::begin(ca);
  static_assert(std::is_same_v<decltype(*ita), internal::AnyAtomicVectorSpaceDescriptor<double>>);
  EXPECT_EQ(get_type_index(*ita), get_type_index(internal::AnyAtomicVectorSpaceDescriptor<double>{Axis{}}));
  EXPECT_TRUE(*ita == Axis{});
  EXPECT_TRUE(*(ita + 1) == angle::Radians{});
  EXPECT_TRUE(ita[1] == angle::Radians{});
  EXPECT_TRUE(*(2 + ita) == Dimensions<3>{});
  EXPECT_TRUE(*(ita + ++std::begin(ca)) == angle::Radians{});
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

  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}} + DynamicDescriptor<double> {angle::Degrees{}, Dimensions<2>{}} ==
    DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Dimensions<2>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Dimensions<2>{}} + DynamicDescriptor<double> {Axis{}, angle::Degrees{}, Dimensions<4>{}} ==
    DynamicDescriptor<double> {Axis{}, angle::Radians{}, Dimensions<3>{}, angle::Degrees{}, Dimensions<4>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}} + Dimensions<2>{} ==
    DynamicDescriptor<double> {Axis{}, angle::Radians{}, Dimensions<2>{}}));
  EXPECT_TRUE((Dimensions<2>{} + DynamicDescriptor<double> {angle::Degrees{}, Axis{}} ==
    DynamicDescriptor<double> {Dimensions<2>{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}} + angle::Degrees{} ==
    DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}}));
  EXPECT_TRUE((StaticDescriptor<angle::Radians>{} + DynamicDescriptor<double> {angle::Degrees{}, Axis{}} ==
    DynamicDescriptor<double> {angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}} + StaticDescriptor<angle::Degrees, Axis>{} ==
    DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((StaticDescriptor<Axis, angle::Radians>{} + DynamicDescriptor<double> {angle::Degrees{}, Axis{}} ==
    DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
}


TEST(descriptors, fixed_replication)
{
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 4>{} == Dimensions<12>{});
  static_assert(std::integral_constant<std::size_t, 4>{} * Dimensions<3>{} == Dimensions<12>{});
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 4>{} * std::integral_constant<std::size_t, 2>{} == Dimensions<24>{});
  static_assert(StaticDescriptor<>{} * std::integral_constant<std::size_t, 4>{} == Dimensions<0>{});
  static_assert(std::integral_constant<std::size_t, 4>{} * StaticDescriptor<>{} == Dimensions<0>{});
  static_assert(Dimensions<3>{} * std::integral_constant<std::size_t, 0>{} == Dimensions<0>{});
  static_assert(std::integral_constant<std::size_t, 0>{} * Dimensions<3>{} == Dimensions<0>{});

  static_assert(Distance{} * std::integral_constant<std::size_t, 1>{} == Distance{});
  static_assert(std::integral_constant<std::size_t, 1>{} * Distance{} == Distance{});
  static_assert(Distance{} * std::integral_constant<std::size_t, 2>{} == StaticDescriptor<Distance, Distance>{});
  static_assert(std::integral_constant<std::size_t, 2>{} * Distance{} == StaticDescriptor<Distance, Distance>{});
  static_assert(StaticDescriptor<Distance, angle::Radians>{} * std::integral_constant<std::size_t, 2>{} == StaticDescriptor<Distance, angle::Radians, Distance, angle::Radians>{});

  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 1>{})), std::tuple<Axis, Distance, Axis>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 2>{})), std::tuple<Axis, Distance, Dimensions<2>, Distance, Axis>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Axis, Distance, Axis>{} * std::integral_constant<std::size_t, 3>{})), std::tuple<Axis, Distance, Dimensions<2>, Distance, Dimensions<2>, Distance, Axis>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Axis, Distance, Dimensions<2>>{} * std::integral_constant<std::size_t, 4>{})), std::tuple<Axis, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<3>, Distance, Dimensions<2>>>);
}


TEST(descriptors, dynamic_replication)
{
  static_assert(Dimensions<3>{} * 4u == Dimensions{12});
  static_assert(4u * Dimensions<3>{} == Dimensions{12});
  static_assert(Dimensions<3>{} * 4u * std::integral_constant<std::size_t, 2>{} == Dimensions{24});
  static_assert(StaticDescriptor<>{} * 4u == Dimensions{0});
  static_assert(4u * StaticDescriptor<>{} == Dimensions{0});
  static_assert(Dimensions<3>{} * 0u == Dimensions{0});
  static_assert(0u * Dimensions<3>{} == Dimensions{0});

  static_assert(Dimensions{3} * 4u == Dimensions{12});
  static_assert(4u * Dimensions{3} == Dimensions{12});
  static_assert(Dimensions{3} * 4u * std::integral_constant<std::size_t, 2>{} == Dimensions{24});
  static_assert(Dimensions{0} * 4u == Dimensions{0});
  static_assert(4u * Dimensions{0} == Dimensions{0});
  static_assert(Dimensions{3} * 0u == Dimensions{0});
  static_assert(0u * Dimensions{3} == Dimensions{0});

  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Distance{}} * std::integral_constant<std::size_t, 2>{} == DynamicDescriptor<double> {Axis{}, Distance{}, Axis{}, Distance{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Distance{}} * 2u == DynamicDescriptor<double> {Axis{}, Distance{}, Axis{}, Distance{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Distance{}, angle::Degrees{}, Axis{}} * 2u == DynamicDescriptor<double> {Axis{}, Distance{}, angle::Degrees{}, Dimensions<2>{}, Distance{}, angle::Degrees{}, Axis{}}));
}