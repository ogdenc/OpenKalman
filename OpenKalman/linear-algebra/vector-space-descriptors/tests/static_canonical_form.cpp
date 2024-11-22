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
 * \brief Tests for \ref descriptor::internal::static_canonical_form_t
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/traits/internal/static_canonical_form.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, static_canonical_form)
{
  using namespace internal;
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_canonical_form_t<Dimensions<0>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Dimensions<0>>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Dimensions<0>, Dimensions<0>>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_canonical_form_t<Axis>, Axis>);
  static_assert(std::is_same_v<static_canonical_form_t<Dimensions<1>>, Axis>);
  static_assert(std::is_same_v<static_canonical_form_t<Dimensions<3>>, Dimensions<3>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Axis, Axis, Axis>>, Dimensions<3>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Dimensions<3>>>, Dimensions<3>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Dimensions<3>, Dimensions<2>>>, Dimensions<5>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<angle::Radians, Dimensions<1>, Dimensions<2>>>, StaticDescriptor<angle::Radians, Dimensions<3>>>);
  static_assert(std::is_same_v<static_canonical_form_t<angle::Radians>, angle::Radians>);
  static_assert(not std::is_same_v<static_canonical_form_t<angle::Degrees>, angle::Radians>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Axis>>, Axis>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Axis, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<StaticDescriptor<Axis>, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<Dimensions<3>, angle::Radians>>, StaticDescriptor<Dimensions<3>, angle::Radians>>);
  static_assert(std::is_same_v<static_canonical_form_t<StaticDescriptor<angle::Radians, Dimensions<3>>>, StaticDescriptor<angle::Radians, Dimensions<3>>>);
  static_assert(std::is_same_v<static_canonical_form_t<Polar<Distance, angle::Radians>>, Polar<Distance, angle::Radians>>);
  static_assert(std::is_same_v<static_canonical_form_t<Spherical<Distance, angle::Radians, inclination::Radians>>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not std::is_same_v<static_canonical_form_t<StaticDescriptor<Axis, angle::Radians, angle::Radians>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not std::is_same_v<static_canonical_form_t<StaticDescriptor<Axis, angle::Radians>>, StaticDescriptor<Polar<Distance, angle::Radians>>>);
}


