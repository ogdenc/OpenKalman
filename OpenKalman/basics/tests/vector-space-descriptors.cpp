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

#include <gtest/gtest.h>
#include "basics/basics.hpp"

using namespace OpenKalman;


TEST(basics, integral_constant)
{
  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 0>> == 0);
  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(dimension_size_of_v<std::integral_constant<int, 3>> == 3);
  static_assert(euclidean_dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(vector_space_component_count_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(get_dimension_size_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_vector_space_descriptor_component_count_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(fixed_vector_space_descriptor<std::integral_constant<std::size_t, 3>>);
  static_assert(fixed_vector_space_descriptor<std::integral_constant<int, 3>>);
  static_assert(not composite_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(not composite_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(atomic_fixed_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(atomic_fixed_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(euclidean_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(euclidean_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
}


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
  static_assert(get_dimension_size_of(Dimensions{FixedDescriptor<Axis, Axis> {}}) == 2);
  static_assert(get_dimension_size_of(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(fixed_vector_space_descriptor<Dimensions<3>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<1>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<2>>);
  static_assert(atomic_fixed_vector_space_descriptor<Dimensions<1>>);
  static_assert(atomic_fixed_vector_space_descriptor<Dimensions<2>>);
  static_assert(euclidean_vector_space_descriptor<Dimensions<1>>);
  static_assert(euclidean_vector_space_descriptor<Dimensions<2>>);
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
  static_assert(not composite_vector_space_descriptor<Axis>);
  static_assert(euclidean_vector_space_descriptor<Axis>);
}


TEST(basics, Distance)
{
  static_assert(dimension_size_of_v<Distance> == 1);
  static_assert(euclidean_dimension_size_of_v<Distance> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Distance>, Axis>);
  static_assert(get_dimension_size_of(Distance{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Distance{}) == 1);
  static_assert(not composite_vector_space_descriptor<Distance>);
  static_assert(fixed_vector_space_descriptor<Distance>);
}


TEST(basics, Angle)
{
  static_assert(dimension_size_of_v<angle::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<angle::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<angle::Radians>, angle::Radians>);
  static_assert(get_dimension_size_of(angle::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(angle::Radians{}) == 2);
  static_assert(not composite_vector_space_descriptor<angle::Radians>);
  static_assert(fixed_vector_space_descriptor<angle::Radians>);
}


TEST(basics, Inclination)
{
  static_assert(dimension_size_of_v<inclination::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<inclination::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<inclination::Radians>, Axis>);
  static_assert(get_dimension_size_of(inclination::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(inclination::Radians{}) == 2);
  static_assert(not composite_vector_space_descriptor<inclination::Radians>);
  static_assert(fixed_vector_space_descriptor<inclination::Radians>);
}


TEST(basics, Polar)
{
  static_assert(dimension_size_of_v<Polar<Distance, angle::Radians>> == 2);
  static_assert(euclidean_dimension_size_of_v<Polar<Distance, angle::Radians>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<Polar<Distance, angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(get_dimension_size_of(Polar<Distance, angle::Radians>{}) == 2);
  static_assert(get_euclidean_dimension_size_of(Polar<Distance, angle::Radians>{}) == 3);
  static_assert(not composite_vector_space_descriptor<Polar<Distance, angle::Radians>>);
  static_assert(fixed_vector_space_descriptor<Polar<Distance, angle::Radians>>);
}


TEST(basics, Spherical)
{
  static_assert(dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 4);
  static_assert(std::is_same_v<dimension_difference_of_t<Spherical<Distance, angle::Radians, inclination::Radians>>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 4);
  static_assert(not composite_vector_space_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(fixed_vector_space_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
}


TEST(basics, FixedDescriptor)
{
  static_assert(dimension_size_of_v<FixedDescriptor<Axis, Axis>> == 2);
  static_assert(euclidean_dimension_size_of_v<FixedDescriptor<Axis, Axis>> == 2);
  static_assert(vector_space_component_count_v<FixedDescriptor<Axis, Axis>> == 2);
  static_assert(dimension_size_of_v<FixedDescriptor<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<FixedDescriptor<Axis, Axis, angle::Radians>> == 4);
  static_assert(vector_space_component_count_v<FixedDescriptor<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_vector_space_descriptor<FixedDescriptor<>>);
  static_assert(euclidean_vector_space_descriptor<FixedDescriptor<Axis, Axis, Axis>>);
  static_assert(euclidean_vector_space_descriptor<FixedDescriptor<FixedDescriptor<Axis>>>);
  static_assert(euclidean_vector_space_descriptor<FixedDescriptor<FixedDescriptor<Axis>, FixedDescriptor<Axis>>>);
  static_assert(not euclidean_vector_space_descriptor<FixedDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_vector_space_descriptor<FixedDescriptor<angle::Radians, Axis, Axis>>);
  static_assert(fixed_vector_space_descriptor<FixedDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not atomic_fixed_vector_space_descriptor<FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<dimension_difference_of_t<FixedDescriptor<Distance, angle::Radians, inclination::Radians>>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(FixedDescriptor<Axis, Axis, angle::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(FixedDescriptor<Axis, Axis, angle::Radians>{}) == 4);
  static_assert(get_vector_space_descriptor_component_count_of(FixedDescriptor<Axis, FixedDescriptor<Axis, angle::Radians>, angle::Radians>{}) == 4);
}


TEST(basics, prepend_append)
{
  static_assert(std::is_same_v<FixedDescriptor<angle::Radians, Axis>::Prepend<Axis>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<FixedDescriptor<Axis, angle::Radians>::Prepend<Axis>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<FixedDescriptor<angle::Radians, Axis>::Append<angle::Radians>, FixedDescriptor<angle::Radians, Axis, angle::Radians>>);
  static_assert(!std::is_same_v<FixedDescriptor<Axis, angle::Radians>::Append<angle::Radians>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians>::Append<Polar<Distance, angle::Radians>>, FixedDescriptor<Axis, angle::Radians, Polar<Distance, angle::Radians>>>);
}


TEST(basics, Take)
{
  static_assert(std::is_same_v<FixedDescriptor<>::Take<0>, FixedDescriptor<>>);
  static_assert(std::is_same_v<FixedDescriptor<angle::Radians>::Take<1>, FixedDescriptor<angle::Radians>>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, FixedDescriptor<Axis, angle::Radians, Axis, Axis>>);
}


TEST(basics, Drop)
{
  static_assert(std::is_same_v<FixedDescriptor<Axis>::Drop<0>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<FixedDescriptor<angle::Radians>::Drop<1>, FixedDescriptor<>>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Drop<3>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis, angle::Radians, Axis>::Drop<3>, FixedDescriptor<angle::Radians, Axis>>);
}


TEST(basics, Select)
{
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis>::Select<0>, Axis>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis>::Select<1>, angle::Radians>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis>::Select<2>, Axis>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis, Polar<>, Distance>::Select<3>, Polar<>>);
  static_assert(std::is_same_v<FixedDescriptor<Axis, angle::Radians, Axis, Polar<>, Distance>::Select<4>, Distance>);
  static_assert(std::is_same_v<FixedDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<0>, Dimensions<3>>);
  static_assert(std::is_same_v<FixedDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<1>, Dimensions<2>>);
  static_assert(std::is_same_v<FixedDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<2>, Axis>);
}


TEST(basics, replicate_fixed_vector_space_descriptor)
{
  static_assert(std::is_same_v<replicate_fixed_vector_space_descriptor_t<angle::Radians, 0>, FixedDescriptor<>>);
  static_assert(std::is_same_v<replicate_fixed_vector_space_descriptor_t<angle::Radians, 1>, angle::Radians>);
  static_assert(std::is_same_v<replicate_fixed_vector_space_descriptor_t<angle::Radians, 2>, FixedDescriptor<angle::Radians, angle::Radians>>);
  static_assert(std::is_same_v<replicate_fixed_vector_space_descriptor_t<FixedDescriptor<angle::Radians, Axis>, 2>, FixedDescriptor<FixedDescriptor<angle::Radians, Axis>, FixedDescriptor<angle::Radians, Axis>>>);
  static_assert(std::is_same_v<replicate_fixed_vector_space_descriptor_t<Dimensions<3>, 2>, FixedDescriptor<Dimensions<3>, Dimensions<3>>>);
}


TEST(basics, concatenate_fixed_vector_space_descriptor_t)
{
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<>, FixedDescriptor<>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<Axis>>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<Axis>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<Axis>, FixedDescriptor<>>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<angle::Radians>, Axis>, FixedDescriptor<angle::Radians, Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<Axis, FixedDescriptor<angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<>, FixedDescriptor<angle::Radians>>, FixedDescriptor<angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<Axis>, FixedDescriptor<angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<angle::Radians, Axis>>, FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t <FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<angle::Radians, Axis>, FixedDescriptor<Axis, angle::Radians>>,
    FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<Axis>, Polar<Distance, angle::Radians>>, FixedDescriptor<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<Polar<Distance, angle::Radians>, FixedDescriptor<Axis>>, FixedDescriptor<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_vector_space_descriptor_t<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>,
    FixedDescriptor<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>>);
}


TEST(basics, canonical_fixed_vector_space_descriptor)
{
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<Dimensions<0>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Dimensions<0>>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Dimensions<0>, Dimensions<0>>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<Axis>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<Dimensions<1>>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<Dimensions<3>>, FixedDescriptor<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Dimensions<3>>>, FixedDescriptor<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Dimensions<3>, Dimensions<2>>>, FixedDescriptor<Axis, Axis, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<angle::Radians, Dimensions<1>, Dimensions<2>>>, FixedDescriptor<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<angle::Radians>, FixedDescriptor<angle::Radians>>);
  static_assert(not std::is_same_v<canonical_fixed_vector_space_descriptor_t<angle::Degrees>, angle::Radians>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Axis>>, FixedDescriptor<Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Axis, angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<FixedDescriptor<Axis>, angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Dimensions<3>, angle::Radians>>, FixedDescriptor<Axis, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<angle::Radians, Dimensions<3>>>, FixedDescriptor<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<Polar<Distance, angle::Radians>>, FixedDescriptor<Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<canonical_fixed_vector_space_descriptor_t<Spherical<Distance, angle::Radians, inclination::Radians>>, FixedDescriptor<Spherical<Distance, angle::Radians, inclination::Radians>>>);
  static_assert(not std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Axis, angle::Radians, angle::Radians>>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not std::is_same_v<canonical_fixed_vector_space_descriptor_t<FixedDescriptor<Axis, angle::Radians>>, FixedDescriptor<Polar<Distance, angle::Radians>>>);
}


TEST(basics, reverse_fixed_vector_space_descriptor)
{
  static_assert(std::is_same_v<reverse_fixed_vector_space_descriptor_t<FixedDescriptor<>>, FixedDescriptor<>>);
  static_assert(std::is_same_v<reverse_fixed_vector_space_descriptor_t<FixedDescriptor<Dimensions<3>, Dimensions<2>>>, FixedDescriptor<Dimensions<2>, Dimensions<3>>>);
  static_assert(std::is_same_v<reverse_fixed_vector_space_descriptor_t<FixedDescriptor<angle::Radians, Dimensions<1>, Dimensions<2>>>, FixedDescriptor<Dimensions<2>, Dimensions<1>, angle::Radians>>);
  static_assert(std::is_same_v<reverse_fixed_vector_space_descriptor_t<FixedDescriptor<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>, FixedDescriptor<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>);
}


TEST(basics, maybe_equivalent_to)
{
  static_assert(maybe_equivalent_to<>);
  static_assert(maybe_equivalent_to<Axis>);
  static_assert(maybe_equivalent_to<FixedDescriptor<>, int>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<dynamic_size>>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<dynamic_size>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<dynamic_size>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<dynamic_size>, Axis, Dimensions<dynamic_size>>);
  static_assert(not maybe_equivalent_to<Axis, Polar<>>);
  static_assert(not maybe_equivalent_to<Polar<>, angle::Radians>);

  static_assert(maybe_equivalent_to<FixedDescriptor<>, Dimensions<0>>);
  static_assert(maybe_equivalent_to<Dimensions<0>, std::size_t>);
  static_assert(maybe_equivalent_to<Dimensions<1>, std::size_t>);
  static_assert(maybe_equivalent_to<Dimensions<dynamic_size>, Dimensions<10>, int, Dimensions<10>>);
  static_assert(not maybe_equivalent_to<Dimensions<dynamic_size>, Dimensions<10>, int, Dimensions<5>>);
  static_assert(not maybe_equivalent_to<angle::Degrees, Dimensions<1>, std::size_t>);
  static_assert(not maybe_equivalent_to<int, angle::Radians>);
  static_assert(not maybe_equivalent_to<angle::Degrees, int>);
}


TEST(basics, equivalent_to)
{
  static_assert(equivalent_to<>);
  static_assert(equivalent_to<Axis>);
  static_assert(equivalent_to<FixedDescriptor<>, FixedDescriptor<>>);
  static_assert(equivalent_to<Dimensions<0>, FixedDescriptor<>>);
  static_assert(equivalent_to<FixedDescriptor<Dimensions<0>>, FixedDescriptor<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(equivalent_to<Dimensions<1>, Axis>);
  static_assert(equivalent_to<FixedDescriptor<Dimensions<1>>, FixedDescriptor<Axis>>);
  static_assert(equivalent_to<Axis, Dimensions<1>>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, FixedDescriptor<Axis>>);
  static_assert(equivalent_to<FixedDescriptor<Axis>, Axis>);
  static_assert(equivalent_to<FixedDescriptor<Axis>, FixedDescriptor<Axis>>);
  static_assert(equivalent_to<FixedDescriptor<Axis, angle::Radians, Axis>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<FixedDescriptor<Dimensions<2>, angle::Radians, Dimensions<3>>, FixedDescriptor<Axis, Axis, angle::Radians, Axis, Axis, Axis>>);
  static_assert(equivalent_to<FixedDescriptor<Axis, angle::Radians, Axis>, FixedDescriptor<Axis, angle::Radians, FixedDescriptor<FixedDescriptor<Axis>>>>);
  static_assert(equivalent_to<FixedDescriptor<FixedDescriptor<Axis>, angle::Radians, FixedDescriptor<Axis>>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<Distance, angle::Radians>, Polar<Distance, angle::Radians>>);
  static_assert(equivalent_to<Spherical<Distance, angle::Radians, inclination::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not equivalent_to<FixedDescriptor<Axis, angle::Radians, angle::Radians>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not equivalent_to<FixedDescriptor<Axis, angle::Radians>, Polar<Distance, angle::Radians>>);

  static_assert(not equivalent_to<Dimensions<0>, std::size_t>);
  static_assert(not equivalent_to<Dimensions<1>, std::size_t>);
  static_assert(not equivalent_to<Dimensions<dynamic_size>, Dimensions<10>, int, Dimensions<10>>);
  static_assert(not equivalent_to<Dimensions<dynamic_size>, Dimensions<10>, int, Dimensions<5>>);
  static_assert(not equivalent_to<angle::Degrees, Dimensions<1>, std::size_t>);
  static_assert(not equivalent_to<int, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, int>);
}


TEST(basics, prefix_of)
{
  using namespace internal;
  static_assert(prefix_of<FixedDescriptor<>, Axis>);
  static_assert(prefix_of<FixedDescriptor<>, Dimensions<2>>);
  static_assert(prefix_of<FixedDescriptor<>, FixedDescriptor<Axis>>);
  static_assert(prefix_of<FixedDescriptor<>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<FixedDescriptor<Axis>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<FixedDescriptor<Axis>, FixedDescriptor<Dimensions<2>, angle::Radians>>);
  static_assert(prefix_of<Axis, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<Axis, FixedDescriptor<Dimensions<2>, angle::Radians>>);
  static_assert(not prefix_of<angle::Radians, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(not prefix_of<FixedDescriptor<angle::Radians>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(prefix_of<FixedDescriptor<Axis, angle::Radians, Axis>, FixedDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<FixedDescriptor<Axis, angle::Radians, angle::Radians>, FixedDescriptor<Axis, angle::Radians, Axis>>);
}


TEST(basics, suffix_of)
{
  using namespace internal;
  static_assert(suffix_of<FixedDescriptor<>, Axis>);
  static_assert(suffix_of<FixedDescriptor<>, Dimensions<2>>);
  static_assert(suffix_of<FixedDescriptor<>, FixedDescriptor<Axis>>);
  static_assert(suffix_of<FixedDescriptor<>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<FixedDescriptor<Axis>, FixedDescriptor<angle::Radians, Axis>>);
  static_assert(suffix_of<FixedDescriptor<Axis>, FixedDescriptor<angle::Radians, Dimensions<2>>>);
  static_assert(suffix_of<Axis, FixedDescriptor<angle::Radians, Axis>>);
  static_assert(suffix_of<Axis, FixedDescriptor<angle::Radians, Dimensions<2>>>);
  static_assert(suffix_of<angle::Radians, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<FixedDescriptor<angle::Radians>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(suffix_of<FixedDescriptor<Axis, angle::Radians, Axis>, FixedDescriptor<Axis, Axis, angle::Radians, Axis>>);
  static_assert(not suffix_of<FixedDescriptor<Axis, angle::Radians, angle::Radians>, FixedDescriptor<Axis, angle::Radians, Axis>>);
}


TEST(basics, base_of)
{
  // either prefix_of or suffix_of
  using namespace internal;
  static_assert(equivalent_to<base_of_t<FixedDescriptor<>, Axis>, Axis>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<>, Dimensions<2>>, Dimensions<2>>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<>, FixedDescriptor<Axis>>, Axis>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<>, FixedDescriptor<Axis, angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<FixedDescriptor<Axis>, angle::Radians>>, FixedDescriptor<>>);

  // prefix_of
  static_assert(equivalent_to<base_of_t<Axis, FixedDescriptor<Axis, angle::Radians>>, angle::Radians>);
  static_assert(equivalent_to<base_of_t<Axis, FixedDescriptor<Dimensions<2>, angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(equivalent_to<base_of_t<Axis, FixedDescriptor<Axis, angle::Radians>>, angle::Radians>);
  static_assert(equivalent_to<base_of_t<Axis, FixedDescriptor<Dimensions<2>, angle::Radians>>, FixedDescriptor<Axis, angle::Radians>>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<Axis, angle::Radians, Axis>>, Axis>);

  // suffix_of
  static_assert(equivalent_to<base_of_t<Axis, FixedDescriptor<angle::Radians, Axis>>, angle::Radians>);
  static_assert(equivalent_to<base_of_t<Axis, FixedDescriptor<angle::Radians, Dimensions<2>>>, FixedDescriptor<angle::Radians, Axis>>);
  static_assert(equivalent_to<base_of_t<angle::Radians, FixedDescriptor<Axis, angle::Radians>>, Axis>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<Axis, angle::Radians>, FixedDescriptor<Axis, Axis, angle::Radians>>, Axis>);
  static_assert(equivalent_to<base_of_t<FixedDescriptor<Axis, angle::Radians, Axis>, FixedDescriptor<Axis, Axis, angle::Radians, Axis>>, Axis>);
}


TEST(basics, uniform_fixed_vector_space_descriptor)
{
  static_assert(not uniform_fixed_vector_space_descriptor<Dimensions<0>>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<Dimensions<1>>, Axis>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<Dimensions<5>>, Axis>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<std::integral_constant<std::size_t, 5>>, Axis>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<std::size_t>, Axis>);

  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<Axis>, Axis>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<Distance>, Distance>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<angle::Radians>, angle::Radians>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<inclination::Radians>, inclination::Radians>);
  static_assert(not uniform_fixed_vector_space_descriptor<Polar<>>);
  static_assert(not uniform_fixed_vector_space_descriptor<Spherical<>>);

  static_assert(not uniform_fixed_vector_space_descriptor<FixedDescriptor<>>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<FixedDescriptor<Axis>>, Axis>);
  static_assert(not uniform_fixed_vector_space_descriptor<FixedDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<FixedDescriptor<Axis, Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_fixed_vector_space_descriptor_component_of_t<FixedDescriptor<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(not uniform_fixed_vector_space_descriptor<FixedDescriptor<Polar<>, Polar<>>>);
}


TEST(basics, comparison)
{
  // Note: some tests cannot be static_assert because of a bug in GCC 10.0
#if OPENKALMAN_CPP_FEATURE_CONCEPTS
  static_assert(Dimensions<3>{} == Dimensions<3>{});
  static_assert(Dimensions<3>{} <= Dimensions<3>{});
  static_assert(Dimensions<3>{} >= Dimensions<3>{});
  static_assert((Dimensions<3>{} != Dimensions<4>{}));
  static_assert((Dimensions<3>{} < Dimensions<4>{}));
  static_assert((Dimensions<3>{} <= Dimensions<4>{}));
  static_assert((Dimensions<4>{} > Dimensions<3>{}));
  static_assert((Dimensions<4>{} >= Dimensions<3>{}));

  static_assert((Dimensions<3>{} == FixedDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} <= FixedDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} <= FixedDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} < FixedDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} >= FixedDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} >= FixedDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} > FixedDescriptor<Axis, Axis, Axis>{}));

  EXPECT_TRUE(FixedDescriptor<> {} == FixedDescriptor<> {});
#else
  EXPECT_TRUE(Dimensions<3>{} == Dimensions<3>{});
  EXPECT_TRUE(Dimensions<3>{} <= Dimensions<3>{});
  EXPECT_TRUE(Dimensions<3>{} >= Dimensions<3>{});
  EXPECT_TRUE((Dimensions<3>{} != Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<3>{} < Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<3>{} <= Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<4>{} > Dimensions<3>{}));
  EXPECT_TRUE((Dimensions<4>{} >= Dimensions<3>{}));

  EXPECT_TRUE((Dimensions<3>{} == FixedDescriptor<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} <= FixedDescriptor<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} <= FixedDescriptor<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} < FixedDescriptor<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} >= FixedDescriptor<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} >= FixedDescriptor<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} > FixedDescriptor<Axis, Axis, Axis>{}));

  EXPECT_TRUE(FixedDescriptor<> {} == FixedDescriptor<> {});
#endif
  static_assert(FixedDescriptor<Axis, angle::Radians>{} == FixedDescriptor<Axis, angle::Radians>{});
  static_assert(FixedDescriptor<Axis, angle::Radians>{} <= FixedDescriptor<Axis, angle::Radians>{});
  static_assert(FixedDescriptor<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{} == FixedDescriptor<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{});
  static_assert(FixedDescriptor<Axis, Dimensions<3>, angle::Radians, Axis, Dimensions<2>>{} < FixedDescriptor<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{});
  static_assert(FixedDescriptor<Axis, angle::Radians>{} < FixedDescriptor<Axis, angle::Radians, Axis>{});
  static_assert(FixedDescriptor<Axis, angle::Radians>{} >= FixedDescriptor<Axis, angle::Radians>{});
  static_assert(FixedDescriptor<Axis, angle::Radians, Axis>{} > FixedDescriptor<Axis, angle::Radians>{});

  static_assert(angle::Radians{} == angle::Radians{});
  static_assert(inclination::Radians{} == inclination::Radians{});
  static_assert(angle::Radians{} != inclination::Radians{});
  static_assert(not (angle::Radians{} < inclination::Radians{}));
  static_assert((Polar<Distance, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions<5>{}));
  static_assert((Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} < Dimensions<5>{}));
}


TEST(basics, assignment)
{
  static_assert(std::is_assignable_v<Dimensions<10>&, Dimensions<10>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, Dimensions<11>>);
  static_assert(std::is_assignable_v<Polar<>&, Polar<>>);
}


TEST(basics, fixed_arithmetic)
{
  static_assert(Dimensions<3>{} + Dimensions<4>{} == Dimensions<7>{});
  static_assert(FixedDescriptor<Axis, Axis>{} + FixedDescriptor<Axis, Axis, Axis>{} == Dimensions<5>{});
  static_assert(Polar<Distance, angle::Radians>{} + Dimensions<2>{} == FixedDescriptor<Polar<Distance, angle::Radians>, Dimensions<2>>{});

  static_assert(Dimensions<7>{} - Dimensions<7>{} == Dimensions<0>{});
  static_assert(Dimensions<7>{} - Dimensions<4>{} == Dimensions<3>{});
  static_assert(FixedDescriptor<Distance, Distance, Distance>{} - Distance{} == FixedDescriptor<Distance, Distance>{});
  static_assert(FixedDescriptor<Axis, angle::Radians, Distance>{} - Distance{} == FixedDescriptor<Axis, angle::Radians>{});
  static_assert(FixedDescriptor<Distance, angle::Radians, Dimensions<3>>{} - Dimensions<2>{} == FixedDescriptor<Distance, angle::Radians, Axis>{});
}


TEST(basics, remove_trailing_1D_descriptors)
{
  using D123 = std::tuple<Dimensions<1>, Dimensions<2>, Dimensions<3>>;
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors()), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}, Dimensions<1>{})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{})), std::tuple<>>);
}


TEST(basics, split_head_tail_fixed)
{
  using namespace internal;

  static_assert(std::is_same_v<split_head_tail_fixed_t<Axis>, std::tuple<Axis, FixedDescriptor<>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<FixedDescriptor<Axis, Distance>>, std::tuple<Axis, Distance>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<FixedDescriptor<Dimensions<2>, Distance, Axis>>, std::tuple<Axis, FixedDescriptor<Axis, Distance, Axis>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<FixedDescriptor<Axis, Polar<Distance, angle::Radians>>>, std::tuple<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<FixedDescriptor<Polar<Distance, angle::Radians>, Axis>>, std::tuple<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<FixedDescriptor<FixedDescriptor<Axis, Distance>, Axis>>, std::tuple<Axis, FixedDescriptor<Distance, Axis>>>);
}


TEST(basics, smallest_vector_space_descriptor_fixed)
{
  static_assert(internal::smallest_vector_space_descriptor(Dimensions<3>{}, Dimensions<4>{}) == Dimensions<3>{});
  static_assert(internal::smallest_vector_space_descriptor(Dimensions<3>{}, angle::Radians{}) == angle::Radians{});
  static_assert(internal::smallest_vector_space_descriptor(Dimensions<1>{}, angle::Radians{}) == Dimensions<1>{});
  static_assert(internal::smallest_vector_space_descriptor(angle::Radians{}, Dimensions<1>{}, angle::Degrees{}) == angle::Radians{});
}


TEST(basics, largest_vector_space_descriptor_fixed)
{
  static_assert(internal::largest_vector_space_descriptor(Dimensions<3>{}, Dimensions<4>{}) == Dimensions<4>{});
  static_assert(internal::largest_vector_space_descriptor(Dimensions<3>{}, angle::Radians{}) == Dimensions<3>{});
  static_assert(internal::largest_vector_space_descriptor(Dimensions<2>{}, Spherical<>{}) == Spherical<>{});
  static_assert(internal::largest_vector_space_descriptor(Dimensions<1>{}, angle::Radians{}) == Dimensions<1>{});
  static_assert(internal::largest_vector_space_descriptor(angle::Radians{}, Dimensions<1>{}, angle::Degrees{}) == angle::Radians{});
}


TEST(basics, fixed_vector_space_descriptor_slice)
{
  using namespace internal;

  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<Dimensions<7>, 0, 7>, Dimensions<7>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<Dimensions<7>, 1, 6>, Dimensions<6>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<Dimensions<7>, 2, 3>, Dimensions<3>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<Dimensions<7>, 2, 0>, Dimensions<0>>);

  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 0, 6>, FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 0, 0>, FixedDescriptor<>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 5>, FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 4>, FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 2>, FixedDescriptor<Axis, Distance>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 1>, FixedDescriptor<Axis>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 0>, FixedDescriptor<>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 4>, FixedDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 3>, FixedDescriptor<Distance, Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 1>, FixedDescriptor<Distance>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 0>, FixedDescriptor<>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 3>, FixedDescriptor<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 2>, FixedDescriptor<Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 0>, FixedDescriptor<>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 5, 1>, FixedDescriptor<Axis>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 5, 0>, FixedDescriptor<>>);
  static_assert(equivalent_to<fixed_vector_space_descriptor_slice_t<FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 6, 0>, FixedDescriptor<>>);
}


TEST(basics, get_vector_space_descriptor_slice_fixed)
{
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 7>{}) == Dimensions<7>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 6>{}) == Dimensions<6>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == Dimensions<3>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == Dimensions<0>{});

  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, 0, 7) == Dimensions<7>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, 1, 6) == Dimensions<6>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, 2, 3) == Dimensions<3>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions<7>{}, 2, 0) == Dimensions<0>{});

  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 6>{}) == FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 0>{}) == FixedDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 5>{}) == FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 4>{}) == FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}) == FixedDescriptor<Axis, Distance>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 1>{}) == FixedDescriptor<Axis>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 0>{}) == FixedDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 4>{}) == FixedDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == FixedDescriptor<Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 1>{}) == FixedDescriptor<Distance>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == FixedDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 3>{}) == FixedDescriptor<Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 2>{}) == FixedDescriptor<Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 0>{}) == FixedDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 1>{}) == FixedDescriptor<Axis>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 0>{}) == FixedDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 6>{}, std::integral_constant<std::size_t, 0>{}) == FixedDescriptor<>{});

  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, std::integral_constant<std::size_t, 6>{}) == FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, 6) == FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, std::integral_constant<std::size_t, 2>{}) == FixedDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, 2) == FixedDescriptor<Axis, Distance>{}));

  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 6) == FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 5) == FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 4) == FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 2) == FixedDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 1) == FixedDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 4) == FixedDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 3) == FixedDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 1) == FixedDescriptor<Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 3) == FixedDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 2) == FixedDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 1) == FixedDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 6, 0) == FixedDescriptor<>{}));
}

