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
 * \brief Tests for descriptor::Angle
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_component_collection.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_size.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_is_euclidean.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, Angle)
{
  static_assert(atomic_vector_space_descriptor<angle::Radians>);
  static_assert(not composite_vector_space_descriptor<angle::Radians>);
  static_assert(static_vector_space_descriptor<angle::Radians>);
  static_assert(vector_space_descriptor<angle::Degrees>);
  static_assert(not euclidean_vector_space_descriptor<angle::Radians>);
  static_assert(std::is_same_v<decltype(internal::get_component_collection(angle::Degrees{})), std::array<angle::Degrees, 1>>);

  static_assert(get_size(angle::Radians{}) == 1);
  static_assert(get_euclidean_size(angle::Radians{}) == 2);
  static_assert(not get_is_euclidean(angle::Radians{}));
  static_assert(dimension_size_of_v<angle::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<angle::Radians> == 2);
  static_assert(vector_space_component_count_v<angle::Radians> == 1);
}

