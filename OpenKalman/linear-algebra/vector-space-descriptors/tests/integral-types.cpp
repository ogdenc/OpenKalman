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
 * \brief Tests for integral coefficient types
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
#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"

using namespace OpenKalman;
using namespace OpenKalman::descriptor;

TEST(descriptors, integral_constant)
{
  static_assert(static_vector_space_descriptor<std::integral_constant<std::size_t, 3>>);
  static_assert(static_vector_space_descriptor<std::integral_constant<int, 3>>);
  static_assert(vector_space_descriptor<std::integral_constant<int, 3>>);
  static_assert(euclidean_vector_space_descriptor<std::integral_constant<std::size_t, 0>>);
  static_assert(euclidean_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(euclidean_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(not composite_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(not composite_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(atomic_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(atomic_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(std::is_same_v<decltype(descriptor::internal::get_component_collection(std::integral_constant<std::size_t, 2>{})), std::array<Dimensions<2>, 1>>);

  static_assert(get_size(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_euclidean_size(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_component_count(std::integral_constant<std::size_t, 3>{}) == 1);
  static_assert(get_is_euclidean(std::integral_constant<std::size_t, 5>{}));

  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 0>> == 0);
  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(dimension_size_of_v<std::integral_constant<int, 3>> == 3);
  static_assert(euclidean_dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(vector_space_component_count_v<std::integral_constant<std::size_t, 3>> == 1);
}


TEST(descriptors, integral)
{
  static_assert(vector_space_descriptor<unsigned>);
  static_assert(dynamic_vector_space_descriptor<unsigned>);
  static_assert(not static_vector_space_descriptor<unsigned>);
  static_assert(euclidean_vector_space_descriptor<unsigned>);
  static_assert(not composite_vector_space_descriptor<unsigned>);
  static_assert(not atomic_vector_space_descriptor<unsigned>);
  static_assert(euclidean_vector_space_descriptor<unsigned>);
  static_assert(dimension_size_of_v<unsigned> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<unsigned> == dynamic_size);
  static_assert(get_size(3u) == 3);
  EXPECT_EQ(get_size(0u), 0);
  EXPECT_EQ(get_size(3u), 3);
  static_assert(get_euclidean_size(3u) == 3);
  EXPECT_EQ(get_euclidean_size(3u), 3);
}
