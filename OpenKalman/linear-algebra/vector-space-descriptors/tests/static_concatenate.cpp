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
 * \brief Tests for descriptor::static_concatenate
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, static_concatenate)
{
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<>, StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<Axis>>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<static_concatenate_t<Axis>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<Axis>, StaticDescriptor<>>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<angle::Radians>, Axis>, StaticDescriptor<angle::Radians, Axis>>);
  static_assert(std::is_same_v<static_concatenate_t<Axis, StaticDescriptor<angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<>, StaticDescriptor<angle::Radians>>, StaticDescriptor<angle::Radians>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<Axis>, StaticDescriptor<angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<angle::Radians, Axis>>, StaticDescriptor<Axis, angle::Radians, angle::Radians, Axis>>);
  static_assert(std::is_same_v<static_concatenate_t <StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians>>,
    StaticDescriptor<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<static_concatenate_t<StaticDescriptor<Axis>, Polar<Distance, angle::Radians>>, StaticDescriptor<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<static_concatenate_t<Polar<Distance, angle::Radians>, StaticDescriptor<Axis>>, StaticDescriptor<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<static_concatenate_t<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>,
    StaticDescriptor<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>>);
}


