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
 * \brief Tests for coordinates::Distance
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
#include "linear-algebra/coordinates/descriptors/Distance.hpp"

using namespace OpenKalman::coordinates;

TEST(coordinates, Distance)
{
  static_assert(descriptor<Distance>);
  static_assert(fixed_pattern<Distance>);
  static_assert(pattern<Distance>);
  static_assert(not euclidean_pattern<Distance>);

  static_assert(get_dimension(Distance{}) == 1);
  static_assert(get_stat_dimension(Distance{}) == 1);
  static_assert(not get_is_euclidean(Distance{}));
  static_assert(dimension_of_v<Distance> == 1);
  static_assert(stat_dimension_of_v<Distance> == 1);
}

