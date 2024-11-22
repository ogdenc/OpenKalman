/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for coefficient types
 */

#include "basics/tests/tests.hpp"

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp"

using namespace OpenKalman::descriptor;

#include "linear-algebra/vector-space-descriptors/traits/internal/static_canonical_form.hpp"

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
  static_assert(atomic_static_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(atomic_static_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"

TEST(descriptors, fixed_Dimensions)
{
  static_assert(static_vector_space_descriptor<Dimensions<3>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<1>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<2>>);
  static_assert(atomic_static_vector_space_descriptor<Dimensions<1>>);
  static_assert(atomic_static_vector_space_descriptor<Dimensions<2>>);
  static_assert(euclidean_vector_space_descriptor<Dimensions<1>>);
  static_assert(euclidean_vector_space_descriptor<Dimensions<2>>);
}


TEST(descriptors, Axis)
{
  static_assert(not composite_vector_space_descriptor<Axis>);
  static_assert(euclidean_vector_space_descriptor<Axis>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp" //

TEST(descriptors, Distance)
{
  static_assert(not composite_vector_space_descriptor<Distance>);
  static_assert(static_vector_space_descriptor<Distance>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp" //

TEST(descriptors, Angle)
{
  static_assert(not composite_vector_space_descriptor<angle::Radians>);
  static_assert(static_vector_space_descriptor<angle::Radians>);
  static_assert(static_vector_space_descriptor<angle::Degrees>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp" //

TEST(descriptors, Inclination)
{
  static_assert(not composite_vector_space_descriptor<inclination::Radians>);
  static_assert(static_vector_space_descriptor<inclination::Radians>);
  static_assert(static_vector_space_descriptor<inclination::Degrees>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp" //

TEST(descriptors, Polar)
{
  static_assert(not composite_vector_space_descriptor<Polar<>>);
  static_assert(static_vector_space_descriptor<Polar<>>);
  static_assert(static_vector_space_descriptor<Polar<Distance, angle::Degrees>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp" //

TEST(descriptors, Spherical)
{
  static_assert(not composite_vector_space_descriptor<Spherical<>>);
  static_assert(static_vector_space_descriptor<Spherical<>>);
  static_assert(static_vector_space_descriptor<Spherical<Distance, angle::Degrees, inclination::Degrees>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp" //

TEST(descriptors, StaticDescriptor)
{
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<Axis, Axis, Axis>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<StaticDescriptor<Axis>>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<StaticDescriptor<Axis>, StaticDescriptor<Axis>>>);
  static_assert(not euclidean_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_vector_space_descriptor<StaticDescriptor<angle::Radians, Axis, Axis>>);
  static_assert(static_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not atomic_static_vector_space_descriptor<StaticDescriptor<Axis>>);
}

