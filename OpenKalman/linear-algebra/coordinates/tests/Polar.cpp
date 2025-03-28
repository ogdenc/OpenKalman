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
 * \brief Tests for coordinate::Polar
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_euclidean_size.hpp"
#include "linear-algebra/coordinates/functions/get_component_count.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/size_of.hpp"
#include "linear-algebra/coordinates/traits/euclidean_size_of.hpp"
#include "linear-algebra/coordinates/traits/component_count_of.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"

using namespace OpenKalman::coordinate;

TEST(coordinates, Polar)
{
  static_assert(descriptor<Polar<>>);
  static_assert(fixed_pattern<Polar<>>);
  static_assert(pattern<Polar<Distance, angle::Degrees>>);
  static_assert(not euclidean_pattern<Polar<>>);

  static_assert(get_size(Polar<>{}) == 2);
  static_assert(get_euclidean_size(Polar<>{}) == 3);
  static_assert(not get_is_euclidean(Polar<>{}));
  static_assert(size_of_v<Polar<>> == 2);
  static_assert(euclidean_size_of_v<Polar<>> == 3);
  static_assert(component_count_of_v<Polar<>> == 1);

  static_assert(std::is_assignable_v<Polar<>&, Polar<>>);
}
