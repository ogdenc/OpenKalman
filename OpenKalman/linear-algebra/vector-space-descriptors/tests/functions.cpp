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


#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_difference_of.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp" //


using namespace OpenKalman::descriptor;

#include "linear-algebra/vector-space-descriptors/traits/internal/static_canonical_form.hpp"

TEST(basics, integral_constant)
{

  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 0>> == 0);
  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(dimension_size_of_v<std::integral_constant<int, 3>> == 3);
  static_assert(euclidean_dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(vector_space_component_count_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<std::integral_constant<std::size_t, 3>>, std::integral_constant<std::size_t, 3>>);

  static_assert(get_dimension_size_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_vector_space_descriptor_component_count_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_vector_space_descriptor_is_euclidean(std::integral_constant<std::size_t, 5>{}));
}

#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"

TEST(basics, fixed_Dimensions)
{
  static_assert(dimension_size_of_v<Dimensions<0>> == 0);
  static_assert(dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(euclidean_dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(vector_space_component_count_v<Dimensions<3>> == 3);
  static_assert(get_dimension_size_of(Dimensions<3>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Dimensions<3>{}) == 3);
  static_assert(get_vector_space_descriptor_component_count_of(Dimensions<3>{}) == 3);
  static_assert(get_dimension_size_of(Dimensions{Axis {}}) == 1);
  static_assert(get_dimension_size_of(Dimensions{StaticDescriptor<Axis, Axis> {}}) == 2);
  static_assert(get_dimension_size_of(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(static_cast<std::integral_constant<int, 3>>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
}


TEST(basics, Axis)
{
  static_assert(dimension_size_of_v<Axis> == 1);
  static_assert(euclidean_dimension_size_of_v<Axis> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Axis>, Axis>);
  static_assert(get_dimension_size_of(Axis{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Axis{}) == 1);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp" //

TEST(basics, Distance)
{
  static_assert(dimension_size_of_v<Distance> == 1);
  static_assert(euclidean_dimension_size_of_v<Distance> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Distance>, Axis>);
  static_assert(get_dimension_size_of(Distance{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Distance{}) == 1);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp" //

TEST(basics, Angle)
{
  static_assert(dimension_size_of_v<angle::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<angle::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<angle::Radians>, angle::Radians>);
  static_assert(get_dimension_size_of(angle::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(angle::Radians{}) == 2);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp" //

TEST(basics, Inclination)
{
  static_assert(dimension_size_of_v<inclination::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<inclination::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<inclination::Radians>, Axis>);
  static_assert(get_dimension_size_of(inclination::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(inclination::Radians{}) == 2);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp" //

TEST(basics, Polar)
{
  static_assert(dimension_size_of_v<Polar<Distance, angle::Radians>> == 2);
  static_assert(euclidean_dimension_size_of_v<Polar<Distance, angle::Radians>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<Polar<Distance, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(get_dimension_size_of(Polar<Distance, angle::Radians>{}) == 2);
  static_assert(get_euclidean_dimension_size_of(Polar<Distance, angle::Radians>{}) == 3);
}

#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp" //

TEST(basics, Spherical)
{
  static_assert(dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 4);
  static_assert(std::is_same_v<dimension_difference_of_t<Spherical<Distance, angle::Radians, inclination::Radians>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 4);
}

#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp" //

TEST(basics, StaticDescriptor)
{
  static_assert(dimension_size_of_v<StaticDescriptor<Axis, Axis>> == 2);
  static_assert(euclidean_dimension_size_of_v<StaticDescriptor<Axis, Axis>> == 2);
  static_assert(vector_space_component_count_v<StaticDescriptor<Axis, Axis>> == 2);
  static_assert(dimension_size_of_v<StaticDescriptor<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<StaticDescriptor<Axis, Axis, angle::Radians>> == 4);
  static_assert(vector_space_component_count_v<StaticDescriptor<Axis, Axis, angle::Radians>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<StaticDescriptor<Distance, angle::Radians, inclination::Radians>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(StaticDescriptor<Axis, Axis, angle::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(StaticDescriptor<Axis, Axis, angle::Radians>{}) == 4);
  static_assert(get_vector_space_descriptor_component_count_of(StaticDescriptor<Axis, StaticDescriptor<Axis, angle::Radians>, angle::Radians>{}) == 4);
}

