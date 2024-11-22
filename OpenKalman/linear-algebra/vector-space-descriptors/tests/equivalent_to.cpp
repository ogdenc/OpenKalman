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
 * \brief Tests for descriptor::maybe_equivalent_to and descriptor::equivalent_to
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, maybe_equivalent_to)
{
  static_assert(maybe_equivalent_to<>);

  static_assert(maybe_equivalent_to<std::integral_constant<std::size_t, 2>, int>);
  static_assert(not maybe_equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>);

  static_assert(maybe_equivalent_to<Axis>);
  static_assert(maybe_equivalent_to<StaticDescriptor<>, int>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<>>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<>, Axis, Dimensions<>>);
  static_assert(not maybe_equivalent_to<Axis, Polar<>>);
  static_assert(not maybe_equivalent_to<Polar<>, angle::Radians>);

  static_assert(maybe_equivalent_to<StaticDescriptor<>, Dimensions<0>>);
  static_assert(maybe_equivalent_to<Dimensions<0>, std::size_t>);
  static_assert(maybe_equivalent_to<Dimensions<1>, std::size_t>);
  static_assert(maybe_equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<10>>);
  static_assert(not maybe_equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<5>>);
  static_assert(not maybe_equivalent_to<angle::Degrees, Dimensions<1>, std::size_t>);
  static_assert(not maybe_equivalent_to<int, angle::Radians>);
  static_assert(not maybe_equivalent_to<angle::Degrees, int>);
}


TEST(descriptors, equivalent_to)
{
  static_assert(equivalent_to<>);

  static_assert(not equivalent_to<std::integral_constant<std::size_t, 2>, int>);
  static_assert(equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>>);
  static_assert(equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(not equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>);

  static_assert(equivalent_to<Axis>);
  static_assert(equivalent_to<StaticDescriptor<>, StaticDescriptor<>>);
  static_assert(equivalent_to<Dimensions<0>, StaticDescriptor<>>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<0>>, StaticDescriptor<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(equivalent_to<Dimensions<1>, Axis>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<1>>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<Axis, Dimensions<1>>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis>, Axis>);
  static_assert(equivalent_to<StaticDescriptor<Axis>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<2>, inclination::Radians, Dimensions<3>>, StaticDescriptor<Axis, Axis, inclination::Radians, Axis, Axis, Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, StaticDescriptor<StaticDescriptor<Axis>>>>);
  static_assert(equivalent_to<StaticDescriptor<StaticDescriptor<Axis>, angle::Radians, StaticDescriptor<Axis>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<>, Polar<>>);
  static_assert(equivalent_to<Spherical<>, Spherical<>>);
  static_assert(not equivalent_to<StaticDescriptor<Axis, angle::Radians, angle::Radians>, StaticDescriptor<Axis, angle::Radians, inclination::Radians>>);
  static_assert(not equivalent_to<StaticDescriptor<Axis, angle::Radians>, Polar<>>);

  static_assert(not equivalent_to<Dimensions<0>, std::size_t>);
  static_assert(not equivalent_to<Dimensions<1>, std::size_t>);
  static_assert(not equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<10>>);
  static_assert(not equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<5>>);
  static_assert(not equivalent_to<angle::Degrees, Dimensions<1>, std::size_t>);
  static_assert(not equivalent_to<int, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, int>);
}
