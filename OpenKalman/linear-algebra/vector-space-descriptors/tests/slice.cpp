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
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_collection.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/internal/prefix_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/remove_trailing_1D_descriptors.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/Slice.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, get_slice_fixed)
{
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 7>{}) == Dimensions<7>{});
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 6>{}) == Dimensions<6>{});
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == Dimensions<3>{});
  static_assert(get_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == Dimensions<0>{});

  static_assert(get_slice<double>(Dimensions<7>{}, 0, 7) == Dimensions<7>{});
  static_assert(get_slice<double>(Dimensions<7>{}, 1, 6) == Dimensions<6>{});
  static_assert(get_slice<double>(Dimensions<7>{}, 2, 3) == Dimensions<3>{});
  static_assert(get_slice<double>(Dimensions<7>{}, 2, 0) == Dimensions<0>{});

  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 6>{}) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 5>{}) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 4>{}) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Axis, Distance>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Axis>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 4>{}) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Distance>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 3>{}) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Polar<Distance, angle::Radians>>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Axis>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 6>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});

  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, std::integral_constant<std::size_t, 6>{}) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, 2) == StaticDescriptor<Axis, Distance>{}));

  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 5) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 4) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 2) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 4) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 3) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 1) == StaticDescriptor<Distance>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 3) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 2) == StaticDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 6, 0) == StaticDescriptor<>{}));
}


TEST(descriptors, slice_vector_space_descriptor_dynamic)
{
  using namespace OpenKalman::descriptor::internal;

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


TEST(descriptors, get_slice_dynamic)
{
  using namespace OpenKalman::descriptor::internal;

  static_assert(get_slice<double>(Dimensions{7}, 0, 7) == Dimensions<7>{});
  static_assert(get_slice<double>(Dimensions{7}, 1, 6) == Dimensions<6>{});
  static_assert(get_slice<double>(Dimensions{7}, 2, 3) == Dimensions<3>{});
  static_assert(get_slice<double>(Dimensions{7}, 2, 0) == Dimensions<0>{});

  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 7)));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, -1, 7)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 5) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 4) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 3)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 2) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 4) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 5)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 3) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 2)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 1) == StaticDescriptor<Distance>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 3) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 2) == StaticDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 1)));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 6, 0) == StaticDescriptor<>{}));
  EXPECT_ANY_THROW((get_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 7, -1)));
}
