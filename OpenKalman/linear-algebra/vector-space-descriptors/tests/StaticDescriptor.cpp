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
 * \brief Tests for descriptor::StaticDescriptor
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
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"

using namespace OpenKalman::descriptor;


TEST(descriptors, StaticDescriptor)
{
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<Axis, Axis, Axis>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<StaticDescriptor<Axis>>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<StaticDescriptor<Axis>, StaticDescriptor<Axis>>>);
  static_assert(not euclidean_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_vector_space_descriptor<StaticDescriptor<angle::Radians, Axis, Axis>>);
  static_assert(static_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(atomic_vector_space_descriptor<StaticDescriptor<Axis>>);
  static_assert(composite_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);

  static_assert(std::is_same_v<decltype(internal::get_component_collection(StaticDescriptor<Dimensions<0>, Dimensions<0>>{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::get_component_collection(StaticDescriptor<Dimensions<0>, Distance>{})), std::tuple<Distance>>);
  static_assert(std::is_same_v<decltype(internal::get_component_collection(StaticDescriptor<Distance, Dimensions<0>>{})), std::tuple<Distance>>);
  static_assert(std::is_same_v<decltype(internal::get_component_collection(StaticDescriptor<Axis, Distance, angle::Degrees, Axis>{})), std::tuple<Axis, Distance, angle::Degrees, Axis>>);

  static_assert(get_size(Dimensions{StaticDescriptor<Axis, Axis> {}}) == 2);
  static_assert(get_size(StaticDescriptor<Axis, Axis, angle::Radians>{}) == 3);
  static_assert(get_euclidean_size(StaticDescriptor<Axis, Axis, angle::Radians>{}) == 4);
  static_assert(get_component_count(StaticDescriptor<Axis, StaticDescriptor<Axis, angle::Radians>, angle::Radians>{}) == 3);
  static_assert(get_is_euclidean(StaticDescriptor<Axis, StaticDescriptor<Axis>, Axis>{}));

  static_assert(dimension_size_of_v<StaticDescriptor<Axis, Axis, angle::Radians, Polar<>, Spherical<>>> == 8);
  static_assert(euclidean_dimension_size_of_v<StaticDescriptor<angle::Radians, Axis, Axis, Polar<>, Spherical<>>> == 11);
  static_assert(vector_space_component_count_v<StaticDescriptor<Axis, Axis, angle::Radians, Axis, Polar<>, Spherical<>>> == 5);
}

