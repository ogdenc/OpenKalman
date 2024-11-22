/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for descriptor::internal::prefix_base_of
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/traits/internal/prefix_base_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, prefix_base_of)
{
  using namespace internal;
  static_assert(std::is_same_v<prefix_base_of_t<StaticDescriptor<>, Axis>, Axis>);
  static_assert(std::is_same_v<prefix_base_of_t<StaticDescriptor<>, Dimensions<2>>, Dimensions<2>>);
  static_assert(std::is_same_v<prefix_base_of_t<StaticDescriptor<>, StaticDescriptor<Axis>>, Axis>);
  static_assert(std::is_same_v<prefix_base_of_t<StaticDescriptor<>, StaticDescriptor<Axis, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<prefix_base_of_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<StaticDescriptor<Axis>, angle::Radians>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<prefix_base_of_t<Axis, StaticDescriptor<Axis, angle::Radians>>, angle::Radians>);
  static_assert(std::is_same_v<prefix_base_of_t<Axis, StaticDescriptor<Dimensions<2>, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<prefix_base_of_t<Axis, StaticDescriptor<Axis, angle::Radians>>, angle::Radians>);
  static_assert(std::is_same_v<prefix_base_of_t<Axis, StaticDescriptor<Dimensions<2>, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<prefix_base_of_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>, Axis>);
}


