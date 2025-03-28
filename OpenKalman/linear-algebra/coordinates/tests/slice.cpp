/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for coefficient types
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern_collection.hpp"
#include "linear-algebra/coordinates/functions/get_component_count.hpp"
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/functions/comparison-operators.hpp"
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/functions/internal/remove_trailing_1D_descriptors.hpp"
#include "linear-algebra/coordinates/descriptors/internal/Slice.hpp"

using namespace OpenKalman::coordinate;

TEST(coordinates, get_slice_fixed)
{
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 7>{}) == Dimensions<7>{});
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 6>{}) == Dimensions<6>{});
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == Dimensions<3>{});
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == Dimensions<0>{});

  static_assert(get_slice<double>(Dimensions<7>{}, 0, 7) == Dimensions<7>{});
  static_assert(get_slice<double>(Dimensions<7>{}, 1, 6) == Dimensions<6>{});
  static_assert(get_slice<double>(Dimensions<7>{}, 2, 3) == Dimensions<3>{});
  static_assert(get_slice<double>(Dimensions<7>{}, 2, 0) == Dimensions<0>{});

  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 6>{}) == std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 0>{}) == std::tuple<>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 5>{}) == std::tuple<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 4>{}) == std::tuple<Axis, Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}) == std::tuple<Axis, Distance>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 1>{}) == std::tuple<Axis>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 0>{}) == std::tuple<>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 4>{}) == std::tuple<Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == std::tuple<Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 1>{}) == std::tuple<Distance>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == std::tuple<>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 3>{}) == std::tuple<Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 2>{}) == std::tuple<Polar<Distance, angle::Radians>>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 0>{}) == std::tuple<>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 1>{}) == std::tuple<Axis>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 0>{}) == std::tuple<>{});
  static_assert(get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 6>{}, std::integral_constant<std::size_t, 0>{}) == std::tuple<>{});

  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, std::integral_constant<std::size_t, 6>{}) == std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, 6) == std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, std::integral_constant<std::size_t, 2>{}) == std::tuple<Axis, Distance>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, 2) == std::tuple<Axis, Distance>{}));

  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 6) == std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 5) == std::tuple<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 4) == std::tuple<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 2) == std::tuple<Axis, Distance>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 1) == std::tuple<Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 4) == std::tuple<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 3) == std::tuple<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 1) == std::tuple<Distance>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 3) == std::tuple<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 2) == std::tuple<Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 1) == std::tuple<Axis>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 6, 0) == std::tuple<>{}));
}


TEST(coordinates, slice_vector_space_descriptor_dynamic)
{
  using namespace OpenKalman::coordinate::internal;

  EXPECT_TRUE((DynamicDescriptor<double> {}.slice(0, 0) == DynamicDescriptor<double>{}));

  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(0, 1) == Axis{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(1, 1) == DynamicDescriptor<double> {angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(1, 2) == DynamicDescriptor<double> {angle::Radians{}, Distance{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(0, 3) == DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 0) == DynamicDescriptor<double>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 1) == angle::Radians{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 2) == DynamicDescriptor<double> {angle::Radians{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 3) == DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 4)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 5) == DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 0) == DynamicDescriptor<double>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 1) == Axis{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 2) == DynamicDescriptor<double> {Axis{}, Distance{}}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 3)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 4) == DynamicDescriptor<double> {Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 0) == DynamicDescriptor<double>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 1) == Distance{}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 2)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 3) == DynamicDescriptor<double> {Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 0) == DynamicDescriptor<double> {}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 1)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 2) == Polar<Distance, angle::Radians>{}));

  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians>{}, Axis{}}.slice(0, 2) == Polar<Distance, angle::Radians>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians>{}, Axis{}}.slice(2, 1) == Axis{}));
}


TEST(coordinates, get_slice_dynamic)
{
  using namespace OpenKalman::coordinate::internal;

  static_assert(get_slice<double>(Dimensions{7}, 0, 7) == Dimensions<7>{});
  static_assert(get_slice<double>(Dimensions{7}, 1, 6) == Dimensions<6>{});
  static_assert(get_slice<double>(Dimensions{7}, 2, 3) == Dimensions<3>{});
  static_assert(get_slice<double>(Dimensions{7}, 2, 0) == Dimensions<0>{});

  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 6) == std::tuple<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 7)));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, -1, 7)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 5) == std::tuple<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 4) == std::tuple<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 3)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 2) == std::tuple<Axis, Distance>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 1) == std::tuple<Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 4) == std::tuple<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 5)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 3) == std::tuple<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 2)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 1) == std::tuple<Distance>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 3) == std::tuple<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 2) == std::tuple<Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 1)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 1) == std::tuple<Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 0) == std::tuple<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 6, 0) == std::tuple<>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 7, -1)));
}
