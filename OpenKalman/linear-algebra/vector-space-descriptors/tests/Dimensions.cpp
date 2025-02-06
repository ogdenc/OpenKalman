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
 * \brief Tests for descriptor::Dimensions
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
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp"

using namespace OpenKalman;
using namespace OpenKalman::descriptor;

TEST(descriptors, Dimensions_fixed)
{
  static_assert(atomic_vector_space_descriptor<Dimensions<3>>);
  static_assert(static_vector_space_descriptor<Dimensions<3>>);
  static_assert(vector_space_descriptor<Dimensions<3>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<1>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<2>>);
  static_assert(not composite_vector_space_descriptor<Axis>);
  static_assert(atomic_vector_space_descriptor<Dimensions<1>>);
  static_assert(atomic_vector_space_descriptor<Dimensions<2>>);
  static_assert(euclidean_vector_space_descriptor<Dimensions<1>>);
  static_assert(euclidean_vector_space_descriptor<Dimensions<2>>);
  static_assert(euclidean_vector_space_descriptor<Axis>);
  static_assert(std::is_same_v<decltype(descriptor::internal::get_component_collection(Dimensions<0>{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(descriptor::internal::get_component_collection(Dimensions<1>{})), std::array<Dimensions<1>, 1>>);
  static_assert(std::is_same_v<decltype(descriptor::internal::get_component_collection(Dimensions<2>{})), std::array<Dimensions<2>, 1>>);
  static_assert(std::is_same_v<decltype(descriptor::internal::get_component_collection(Axis{})), std::array<Dimensions<1>, 1>>);

  static_assert(get_size(Dimensions<3>{}) == 3);
  static_assert(get_size(Axis{}) == 1);
  static_assert(get_euclidean_size(Dimensions<3>{}) == 3);
  static_assert(get_euclidean_size(Axis{}) == 1);
  static_assert(get_component_count(Dimensions<3>{}) == 1);
  static_assert(get_size(Dimensions{Axis {}}) == 1);
  static_assert(get_size(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(get_is_euclidean(Dimensions<3>{}));

  static_assert(dimension_size_of_v<Dimensions<0>> == 0);
  static_assert(dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(dimension_size_of_v<Axis> == 1);
  static_assert(euclidean_dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(euclidean_dimension_size_of_v<Axis> == 1);
  static_assert(vector_space_component_count_v<Dimensions<3>> == 1);

  static_assert(static_cast<std::integral_constant<int, 3>>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
}


TEST(descriptors, Dimensions_dynamic)
{
  using D = Dimensions<>;

  static_assert(vector_space_descriptor<D>);
  static_assert(dynamic_vector_space_descriptor<D>);
  static_assert(not static_vector_space_descriptor<D>);
  static_assert(euclidean_vector_space_descriptor<D>);
  static_assert(not composite_vector_space_descriptor<D>);
  static_assert(euclidean_vector_space_descriptor<D>);
  static_assert(not atomic_vector_space_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
  static_assert(get_size(D {0}) == 0);
  static_assert(get_size(D {3}) == 3);
  static_assert(get_size(Dimensions {0}) == 0);
  static_assert(get_size(Dimensions {3}) == 3);
  static_assert(get_euclidean_size(Dimensions{3}) == 3);
  static_assert(get_size(Dimensions<dynamic_size> {Axis {}}) == 1);
  static_assert(get_size(Dimensions<dynamic_size> {Dimensions<3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions {3}) == 3);
}


TEST(descriptors, Dimensions_assignment)
{
  static_assert(std::is_assignable_v<Dimensions<10>&, Dimensions<10>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, Dimensions<11>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, Dimensions<>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, std::size_t>);

  using D = Dimensions<>;

  static_assert(std::is_assignable_v<D&, D>);
  static_assert(std::is_assignable_v<D&, Dimensions<10>>);
  static_assert(std::is_assignable_v<D&, std::integral_constant<int, 3>>);
  static_assert(std::is_assignable_v<D&, Dimensions<11>>);
  static_assert(std::is_assignable_v<D&, DynamicDescriptor<double>>);
  static_assert(std::is_assignable_v<D&, std::size_t>);

  Dimensions d {5u};
  EXPECT_EQ(get_size(d), 5);
  d = 6u;
  EXPECT_EQ(get_size(d), 6);
  d = Dimensions<7>{};
  EXPECT_EQ(get_size(d), 7);
  d = DynamicDescriptor<double> {Dimensions<3>{}, Dimensions<5>{}};
  EXPECT_EQ(get_size(d), 8);
}
