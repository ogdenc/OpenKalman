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
 * \brief Tests for coordinates::Inclination
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"

using namespace OpenKalman::coordinates;

TEST(coordinates, Inclination)
{
  static_assert(descriptor<inclination::Radians>);
  static_assert(fixed_pattern<inclination::Radians>);
  static_assert(pattern<inclination::Degrees>);
  static_assert(not euclidean_pattern<inclination::Radians>);

  static_assert(get_dimension(inclination::Radians{}) == 1);
  static_assert(get_stat_dimension(inclination::Radians{}) == 2);
  static_assert(not get_is_euclidean(inclination::Radians{}));
  static_assert(dimension_of_v<inclination::Radians> == 1);
  static_assert(stat_dimension_of_v<inclination::Radians> == 2);
}

