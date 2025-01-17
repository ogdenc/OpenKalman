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

#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"

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
  static_assert(std::is_same_v<decltype(get_collection_of(std::integral_constant<std::size_t, 2>{})), std::array<Dimensions<2>, 1>>);
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
  static_assert(std::is_same_v<decltype(get_collection_of(Dimensions<0>{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(get_collection_of(Dimensions<1>{})), std::array<Dimensions<1>, 1>>);
  static_assert(std::is_same_v<decltype(get_collection_of(Dimensions<2>{})), std::array<Dimensions<2>, 1>>);
}


TEST(descriptors, Axis)
{
  static_assert(not composite_vector_space_descriptor<Axis>);
  static_assert(euclidean_vector_space_descriptor<Axis>);
  static_assert(std::is_same_v<decltype(get_collection_of(Axis{})), std::array<Dimensions<1>, 1>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp" //

TEST(descriptors, Distance)
{
  static_assert(not composite_vector_space_descriptor<Distance>);
  static_assert(static_vector_space_descriptor<Distance>);
  static_assert(std::is_same_v<decltype(get_collection_of(Distance{})), std::array<Distance, 1>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp" //

TEST(descriptors, Angle)
{
  static_assert(not composite_vector_space_descriptor<angle::Radians>);
  static_assert(static_vector_space_descriptor<angle::Radians>);
  static_assert(static_vector_space_descriptor<angle::Degrees>);
  static_assert(std::is_same_v<decltype(get_collection_of(angle::Degrees{})), std::array<angle::PositiveDegrees, 1>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp" //

TEST(descriptors, Inclination)
{
  static_assert(not composite_vector_space_descriptor<inclination::Radians>);
  static_assert(static_vector_space_descriptor<inclination::Radians>);
  static_assert(static_vector_space_descriptor<inclination::Degrees>);
  static_assert(std::is_same_v<decltype(get_collection_of(inclination::Degrees{})), std::array<inclination::Degrees, 1>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp" //

TEST(descriptors, Polar)
{
  static_assert(not composite_vector_space_descriptor<Polar<>>);
  static_assert(static_vector_space_descriptor<Polar<>>);
  static_assert(static_vector_space_descriptor<Polar<Distance, angle::Degrees>>);
  static_assert(std::is_same_v<decltype(get_collection_of(Polar<Distance, angle::Degrees>{})), std::array<Polar<Distance, angle::PositiveDegrees>, 1>>);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp" //

TEST(descriptors, Spherical)
{
  static_assert(not composite_vector_space_descriptor<Spherical<>>);
  static_assert(static_vector_space_descriptor<Spherical<>>);
  static_assert(static_vector_space_descriptor<Spherical<Distance, angle::Degrees, inclination::Degrees>>);
  static_assert(std::is_same_v<decltype(get_collection_of(Spherical<Distance, angle::Degrees, inclination::Degrees>{})), std::array<Spherical<Distance, angle::PositiveDegrees, inclination::Degrees>, 1>>);
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
  static_assert(atomic_static_vector_space_descriptor<StaticDescriptor<Axis>>);

  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Dimensions<0>, Dimensions<0>>{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Dimensions<0>, Distance>{})), std::tuple<Distance>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Distance, Dimensions<0>>{})), std::tuple<Distance>>);
  static_assert(std::is_same_v<decltype(get_collection_of(StaticDescriptor<Axis, Distance, angle::Degrees, Axis>{})), std::tuple<Axis, Distance, angle::PositiveDegrees, Axis>>);
}

