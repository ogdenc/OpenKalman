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
 * \brief Tests for descriptor::static_reverse
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/traits/static_reverse.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, static_reverse)
{
  static_assert(std::is_same_v<static_reverse_t<StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<static_reverse_t<StaticDescriptor<Dimensions<3>, Dimensions<2>>>, StaticDescriptor<Dimensions<2>, Dimensions<3>>>);
  static_assert(std::is_same_v<static_reverse_t<StaticDescriptor<angle::Radians, Dimensions<1>, Dimensions<2>>>, StaticDescriptor<Dimensions<2>, Dimensions<1>, angle::Radians>>);
  static_assert(std::is_same_v<static_reverse_t<StaticDescriptor<angle::Radians, Dimensions<1>, Polar<Distance, angle::Radians>, Dimensions<2>>>, StaticDescriptor<Dimensions<2>, Polar<Distance, angle::Radians>, Dimensions<1>, angle::Radians>>);
  static_assert(std::is_same_v<static_reverse_t<StaticDescriptor<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>, StaticDescriptor<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>);
}


