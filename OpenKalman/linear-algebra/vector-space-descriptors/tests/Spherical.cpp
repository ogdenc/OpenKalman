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
 * \brief Tests for descriptor::Spherical
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

#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"

using namespace OpenKalman::descriptor;

TEST(descriptors, Spherical)
{
  static_assert(atomic_vector_space_descriptor<Spherical<>>);
  static_assert(not composite_vector_space_descriptor<Spherical<>>);
  static_assert(static_vector_space_descriptor<Spherical<>>);
  static_assert(vector_space_descriptor<Spherical<Distance, angle::Degrees, inclination::Degrees>>);
  static_assert(not euclidean_vector_space_descriptor<Spherical<>>);
  static_assert(std::is_same_v<decltype(internal::get_component_collection(Spherical<Distance, angle::Degrees, inclination::Degrees>{})), std::array<Spherical<Distance, angle::Degrees, inclination::Degrees>, 1>>);

  static_assert(get_size(Spherical<>{}) == 3);
  static_assert(get_euclidean_size(Spherical<>{}) == 4);
  static_assert(not get_is_euclidean(Spherical<>{}));
  static_assert(dimension_size_of_v<Spherical<>> == 3);
  static_assert(euclidean_dimension_size_of_v<Spherical<>> == 4);
  static_assert(vector_space_component_count_v<Spherical<>> == 1);
}
