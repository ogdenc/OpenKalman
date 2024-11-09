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
  static_assert(static_vector_space_descriptor<std::integral_constant<std::size_t, 3>>);
  static_assert(static_vector_space_descriptor<std::integral_constant<int, 3>>);
  static_assert(not composite_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(not composite_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(atomic_static_vector_space_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(atomic_static_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
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
  static_assert(get_dimension_size_of(Dimensions{StaticDescriptor<Axis, Axis> {}}) == 2);
  static_assert(get_dimension_size_of(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(static_vector_space_descriptor<Dimensions<3>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<1>>);
  static_assert(not composite_vector_space_descriptor<Dimensions<2>>);
  static_assert(atomic_static_vector_space_descriptor<Dimensions<1>>);
  static_assert(atomic_static_vector_space_descriptor<Dimensions<2>>);
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
  static_assert(static_vector_space_descriptor<Distance>);
}


TEST(basics, Angle)
{
  static_assert(dimension_size_of_v<angle::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<angle::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<angle::Radians>, angle::Radians>);
  static_assert(get_dimension_size_of(angle::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(angle::Radians{}) == 2);
  static_assert(not composite_vector_space_descriptor<angle::Radians>);
  static_assert(static_vector_space_descriptor<angle::Radians>);
}


TEST(basics, Inclination)
{
  static_assert(dimension_size_of_v<inclination::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<inclination::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<inclination::Radians>, Axis>);
  static_assert(get_dimension_size_of(inclination::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(inclination::Radians{}) == 2);
  static_assert(not composite_vector_space_descriptor<inclination::Radians>);
  static_assert(static_vector_space_descriptor<inclination::Radians>);
}


TEST(basics, Polar)
{
  static_assert(dimension_size_of_v<Polar<Distance, angle::Radians>> == 2);
  static_assert(euclidean_dimension_size_of_v<Polar<Distance, angle::Radians>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<Polar<Distance, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(get_dimension_size_of(Polar<Distance, angle::Radians>{}) == 2);
  static_assert(get_euclidean_dimension_size_of(Polar<Distance, angle::Radians>{}) == 3);
  static_assert(not composite_vector_space_descriptor<Polar<Distance, angle::Radians>>);
  static_assert(static_vector_space_descriptor<Polar<Distance, angle::Radians>>);
}


TEST(basics, Spherical)
{
  static_assert(dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 4);
  static_assert(std::is_same_v<dimension_difference_of_t<Spherical<Distance, angle::Radians, inclination::Radians>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 4);
  static_assert(not composite_vector_space_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(static_vector_space_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
}


TEST(basics, StaticDescriptor)
{
  static_assert(dimension_size_of_v<StaticDescriptor<Axis, Axis>> == 2);
  static_assert(euclidean_dimension_size_of_v<StaticDescriptor<Axis, Axis>> == 2);
  static_assert(vector_space_component_count_v<StaticDescriptor<Axis, Axis>> == 2);
  static_assert(dimension_size_of_v<StaticDescriptor<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<StaticDescriptor<Axis, Axis, angle::Radians>> == 4);
  static_assert(vector_space_component_count_v<StaticDescriptor<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<Axis, Axis, Axis>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<StaticDescriptor<Axis>>>);
  static_assert(euclidean_vector_space_descriptor<StaticDescriptor<StaticDescriptor<Axis>, StaticDescriptor<Axis>>>);
  static_assert(not euclidean_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_vector_space_descriptor<StaticDescriptor<angle::Radians, Axis, Axis>>);
  static_assert(static_vector_space_descriptor<StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(not atomic_static_vector_space_descriptor<StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<dimension_difference_of_t<StaticDescriptor<Distance, angle::Radians, inclination::Radians>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(StaticDescriptor<Axis, Axis, angle::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(StaticDescriptor<Axis, Axis, angle::Radians>{}) == 4);
  static_assert(get_vector_space_descriptor_component_count_of(StaticDescriptor<Axis, StaticDescriptor<Axis, angle::Radians>, angle::Radians>{}) == 4);
}


TEST(basics, prepend_append)
{
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians, Axis>::Prepend<Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<StaticDescriptor<Axis, angle::Radians>::Prepend<Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians, Axis>::Append<angle::Radians>, StaticDescriptor<angle::Radians, Axis, angle::Radians>>);
  static_assert(!std::is_same_v<StaticDescriptor<Axis, angle::Radians>::Append<angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians>::Append<Polar<Distance, angle::Radians>>, StaticDescriptor<Axis, angle::Radians, Polar<Distance, angle::Radians>>>);
}


TEST(basics, Take)
{
  static_assert(std::is_same_v<StaticDescriptor<>::Take<0>, StaticDescriptor<>>);
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians>::Take<1>, StaticDescriptor<angle::Radians>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, StaticDescriptor<Axis, angle::Radians, Axis, Axis>>);
}


TEST(basics, Drop)
{
  static_assert(std::is_same_v<StaticDescriptor<Axis>::Drop<0>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians>::Drop<1>, StaticDescriptor<>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Drop<3>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, angle::Radians, Axis>::Drop<3>, StaticDescriptor<angle::Radians, Axis>>);
}


TEST(basics, Select)
{
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis>::Select<0>, Axis>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis>::Select<1>, angle::Radians>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis>::Select<2>, Axis>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Polar<>, Distance>::Select<3>, Polar<>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Polar<>, Distance>::Select<4>, Distance>);
  static_assert(std::is_same_v<StaticDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<0>, Dimensions<3>>);
  static_assert(std::is_same_v<StaticDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<1>, Dimensions<2>>);
  static_assert(std::is_same_v<StaticDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<2>, Axis>);
}


TEST(basics, replicate_static_vector_space_descriptor)
{
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<angle::Radians, 0>, StaticDescriptor<>>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<angle::Radians, 1>, angle::Radians>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<angle::Radians, 2>, StaticDescriptor<angle::Radians, angle::Radians>>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians, Axis>, 2>, StaticDescriptor<StaticDescriptor<angle::Radians, Axis>, StaticDescriptor<angle::Radians, Axis>>>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<Dimensions<3>, 2>, StaticDescriptor<Dimensions<3>, Dimensions<3>>>);
}


TEST(basics, concatenate_static_vector_space_descriptor_t)
{
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<>, StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<Axis>>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<Axis>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<Axis>, StaticDescriptor<>>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians>, Axis>, StaticDescriptor<angle::Radians, Axis>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<Axis, StaticDescriptor<angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<>, StaticDescriptor<angle::Radians>>, StaticDescriptor<angle::Radians>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<Axis>, StaticDescriptor<angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<angle::Radians, Axis>>, StaticDescriptor<Axis, angle::Radians, angle::Radians, Axis>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t <StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians>>,
    StaticDescriptor<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<StaticDescriptor<Axis>, Polar<Distance, angle::Radians>>, StaticDescriptor<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<Polar<Distance, angle::Radians>, StaticDescriptor<Axis>>, StaticDescriptor<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<concatenate_static_vector_space_descriptor_t<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>,
    StaticDescriptor<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>>);
}


TEST(basics, canonical_static_vector_space_descriptor)
{
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<Dimensions<0>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Dimensions<0>>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Dimensions<0>, Dimensions<0>>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<Axis>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<Dimensions<1>>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<Dimensions<3>>, StaticDescriptor<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Dimensions<3>>>, StaticDescriptor<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Dimensions<3>, Dimensions<2>>>, StaticDescriptor<Axis, Axis, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians, Dimensions<1>, Dimensions<2>>>, StaticDescriptor<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<angle::Radians>, StaticDescriptor<angle::Radians>>);
  static_assert(not std::is_same_v<canonical_static_vector_space_descriptor_t<angle::Degrees>, angle::Radians>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Axis>>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Axis, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<StaticDescriptor<Axis>, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Dimensions<3>, angle::Radians>>, StaticDescriptor<Axis, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians, Dimensions<3>>>, StaticDescriptor<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<Polar<Distance, angle::Radians>>, StaticDescriptor<Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<canonical_static_vector_space_descriptor_t<Spherical<Distance, angle::Radians, inclination::Radians>>, StaticDescriptor<Spherical<Distance, angle::Radians, inclination::Radians>>>);
  static_assert(not std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Axis, angle::Radians, angle::Radians>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not std::is_same_v<canonical_static_vector_space_descriptor_t<StaticDescriptor<Axis, angle::Radians>>, StaticDescriptor<Polar<Distance, angle::Radians>>>);
}


TEST(basics, reverse_static_vector_space_descriptor)
{
  static_assert(std::is_same_v<reverse_static_vector_space_descriptor_t<StaticDescriptor<>>, StaticDescriptor<>>);
  static_assert(std::is_same_v<reverse_static_vector_space_descriptor_t<StaticDescriptor<Dimensions<3>, Dimensions<2>>>, StaticDescriptor<Dimensions<2>, Dimensions<3>>>);
  static_assert(std::is_same_v<reverse_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians, Dimensions<1>, Dimensions<2>>>, StaticDescriptor<Dimensions<2>, Dimensions<1>, angle::Radians>>);
  static_assert(std::is_same_v<reverse_static_vector_space_descriptor_t<StaticDescriptor<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>, StaticDescriptor<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>);
}


TEST(basics, maybe_equivalent_to)
{
  static_assert(maybe_equivalent_to<>);
  static_assert(maybe_equivalent_to<Axis>);
  static_assert(maybe_equivalent_to<StaticDescriptor<>, int>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<dynamic_size>>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<dynamic_size>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<dynamic_size>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<dynamic_size>, Axis, Dimensions<dynamic_size>>);
  static_assert(not maybe_equivalent_to<Axis, Polar<>>);
  static_assert(not maybe_equivalent_to<Polar<>, angle::Radians>);

  static_assert(maybe_equivalent_to<StaticDescriptor<>, Dimensions<0>>);
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
  static_assert(equivalent_to<StaticDescriptor<>, StaticDescriptor<>>);
  static_assert(equivalent_to<Dimensions<0>, StaticDescriptor<>>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<0>>, StaticDescriptor<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(equivalent_to<Dimensions<1>, Axis>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<1>>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<Axis, Dimensions<1>>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis>, Axis>);
  static_assert(equivalent_to<StaticDescriptor<Axis>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<2>, angle::Radians, Dimensions<3>>, StaticDescriptor<Axis, Axis, angle::Radians, Axis, Axis, Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, StaticDescriptor<StaticDescriptor<Axis>>>>);
  static_assert(equivalent_to<StaticDescriptor<StaticDescriptor<Axis>, angle::Radians, StaticDescriptor<Axis>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<Distance, angle::Radians>, Polar<Distance, angle::Radians>>);
  static_assert(equivalent_to<Spherical<Distance, angle::Radians, inclination::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not equivalent_to<StaticDescriptor<Axis, angle::Radians, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not equivalent_to<StaticDescriptor<Axis, angle::Radians>, Polar<Distance, angle::Radians>>);

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
  static_assert(prefix_of<StaticDescriptor<>, Axis>);
  static_assert(prefix_of<StaticDescriptor<>, Dimensions<2>>);
  static_assert(prefix_of<StaticDescriptor<>, StaticDescriptor<Axis>>);
  static_assert(prefix_of<StaticDescriptor<>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis>, StaticDescriptor<Dimensions<2>, angle::Radians>>);
  static_assert(prefix_of<Axis, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<Axis, StaticDescriptor<Dimensions<2>, angle::Radians>>);
  static_assert(not prefix_of<angle::Radians, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(not prefix_of<StaticDescriptor<angle::Radians>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(prefix_of<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<StaticDescriptor<Axis, angle::Radians, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
}


TEST(basics, suffix_of)
{
  using namespace internal;
  static_assert(suffix_of<StaticDescriptor<>, Axis>);
  static_assert(suffix_of<StaticDescriptor<>, Dimensions<2>>);
  static_assert(suffix_of<StaticDescriptor<>, StaticDescriptor<Axis>>);
  static_assert(suffix_of<StaticDescriptor<>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<StaticDescriptor<Axis>, StaticDescriptor<angle::Radians, Axis>>);
  static_assert(suffix_of<StaticDescriptor<Axis>, StaticDescriptor<angle::Radians, Dimensions<2>>>);
  static_assert(suffix_of<Axis, StaticDescriptor<angle::Radians, Axis>>);
  static_assert(suffix_of<Axis, StaticDescriptor<angle::Radians, Dimensions<2>>>);
  static_assert(suffix_of<angle::Radians, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<StaticDescriptor<angle::Radians>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(suffix_of<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(suffix_of<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, Axis, angle::Radians, Axis>>);
  static_assert(not suffix_of<StaticDescriptor<Axis, angle::Radians, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
}


TEST(basics, base_of)
{
  // either prefix_of or suffix_of
  using namespace internal;
  static_assert(equivalent_to<base_of_t<StaticDescriptor<>, Axis>, Axis>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<>, Dimensions<2>>, Dimensions<2>>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<>, StaticDescriptor<Axis>>, Axis>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<>, StaticDescriptor<Axis, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<StaticDescriptor<Axis>, angle::Radians>>, StaticDescriptor<>>);

  // prefix_of
  static_assert(equivalent_to<base_of_t<Axis, StaticDescriptor<Axis, angle::Radians>>, angle::Radians>);
  static_assert(equivalent_to<base_of_t<Axis, StaticDescriptor<Dimensions<2>, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(equivalent_to<base_of_t<Axis, StaticDescriptor<Axis, angle::Radians>>, angle::Radians>);
  static_assert(equivalent_to<base_of_t<Axis, StaticDescriptor<Dimensions<2>, angle::Radians>>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>, Axis>);

  // suffix_of
  static_assert(equivalent_to<base_of_t<Axis, StaticDescriptor<angle::Radians, Axis>>, angle::Radians>);
  static_assert(equivalent_to<base_of_t<Axis, StaticDescriptor<angle::Radians, Dimensions<2>>>, StaticDescriptor<angle::Radians, Axis>>);
  static_assert(equivalent_to<base_of_t<angle::Radians, StaticDescriptor<Axis, angle::Radians>>, Axis>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, Axis, angle::Radians>>, Axis>);
  static_assert(equivalent_to<base_of_t<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, Axis, angle::Radians, Axis>>, Axis>);
}


TEST(basics, uniform_static_vector_space_descriptor)
{
  static_assert(not uniform_static_vector_space_descriptor<Dimensions<0>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Dimensions<1>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Dimensions<5>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::integral_constant<std::size_t, 5>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::size_t>, Axis>);

  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Axis>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Distance>, Distance>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<angle::Radians>, angle::Radians>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<inclination::Radians>, inclination::Radians>);
  static_assert(not uniform_static_vector_space_descriptor<Polar<>>);
  static_assert(not uniform_static_vector_space_descriptor<Spherical<>>);

  static_assert(not uniform_static_vector_space_descriptor<StaticDescriptor<>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<StaticDescriptor<Axis>>, Axis>);
  static_assert(not uniform_static_vector_space_descriptor<StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<StaticDescriptor<Axis, Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<StaticDescriptor<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(not uniform_static_vector_space_descriptor<StaticDescriptor<Polar<>, Polar<>>>);
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

  static_assert((Dimensions<3>{} == StaticDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} <= StaticDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} <= StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} < StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} >= StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} >= StaticDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} > StaticDescriptor<Axis, Axis, Axis>{}));

  EXPECT_TRUE(StaticDescriptor<> {} == StaticDescriptor<> {});
#else
  EXPECT_TRUE(Dimensions<3>{} == Dimensions<3>{});
  EXPECT_TRUE(Dimensions<3>{} <= Dimensions<3>{});
  EXPECT_TRUE(Dimensions<3>{} >= Dimensions<3>{});
  EXPECT_TRUE((Dimensions<3>{} != Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<3>{} < Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<3>{} <= Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<4>{} > Dimensions<3>{}));
  EXPECT_TRUE((Dimensions<4>{} >= Dimensions<3>{}));

  EXPECT_TRUE((Dimensions<3>{} == StaticDescriptor<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} <= StaticDescriptor<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} <= StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} < StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} >= StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} >= StaticDescriptor<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} > StaticDescriptor<Axis, Axis, Axis>{}));

  EXPECT_TRUE(StaticDescriptor<> {} == StaticDescriptor<> {});
#endif
  static_assert(StaticDescriptor<Axis, angle::Radians>{} == StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} <= StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{} == StaticDescriptor<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{});
  static_assert(StaticDescriptor<Axis, Dimensions<3>, angle::Radians, Axis, Dimensions<2>>{} < StaticDescriptor<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} < StaticDescriptor<Axis, angle::Radians, Axis>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} >= StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, angle::Radians, Axis>{} > StaticDescriptor<Axis, angle::Radians>{});

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
  static_assert(StaticDescriptor<Axis, Axis>{} + StaticDescriptor<Axis, Axis, Axis>{} == Dimensions<5>{});
  static_assert(Polar<Distance, angle::Radians>{} + Dimensions<2>{} == StaticDescriptor<Polar<Distance, angle::Radians>, Dimensions<2>>{});

  static_assert(Dimensions<7>{} - Dimensions<7>{} == Dimensions<0>{});
  static_assert(Dimensions<7>{} - Dimensions<4>{} == Dimensions<3>{});
  static_assert(StaticDescriptor<Distance, Distance, Distance>{} - Distance{} == StaticDescriptor<Distance, Distance>{});
  static_assert(StaticDescriptor<Axis, angle::Radians, Distance>{} - Distance{} == StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Distance, angle::Radians, Dimensions<3>>{} - Dimensions<2>{} == StaticDescriptor<Distance, angle::Radians, Axis>{});
}


TEST(basics, remove_trailing_1D_descriptors)
{
  using D123 = std::tuple<Dimensions<1>, Dimensions<2>, Dimensions<3>>;
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}, Dimensions<1>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{}})), std::tuple<>>);
}


TEST(basics, split_head_tail_fixed)
{
  using namespace internal;

  static_assert(std::is_same_v<split_head_tail_fixed_t<Axis>, std::tuple<Axis, StaticDescriptor<>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Axis, Distance>>, std::tuple<Axis, Distance>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Dimensions<2>, Distance, Axis>>, std::tuple<Axis, StaticDescriptor<Axis, Distance, Axis>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Axis, Polar<Distance, angle::Radians>>>, std::tuple<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Polar<Distance, angle::Radians>, Axis>>, std::tuple<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<StaticDescriptor<Axis, Distance>, Axis>>, std::tuple<Axis, StaticDescriptor<Distance, Axis>>>);
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


TEST(basics, static_vector_space_descriptor_slice)
{
  using namespace internal;

  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 0, 7>, Dimensions<7>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 1, 6>, Dimensions<6>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 2, 3>, Dimensions<3>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 2, 0>, Dimensions<0>>);

  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 0, 6>, StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 0, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 5>, StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 4>, StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 2>, StaticDescriptor<Axis, Distance>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 1>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 4>, StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 3>, StaticDescriptor<Distance, Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 1>, StaticDescriptor<Distance>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 3>, StaticDescriptor<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 2>, StaticDescriptor<Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 5, 1>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 5, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 6, 0>, StaticDescriptor<>>);
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

  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 6>{}) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 5>{}) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 4>{}) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Axis, Distance>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Axis>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 4>{}) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Distance>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 3>{}) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Axis>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 6>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});

  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, std::integral_constant<std::size_t, 6>{}) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, 2) == StaticDescriptor<Axis, Distance>{}));

  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 5) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 4) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 2) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 4) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 3) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 1) == StaticDescriptor<Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 3) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 2) == StaticDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 6, 0) == StaticDescriptor<>{}));
}


TEST(basics, vector_space_descriptor_collection)
{
  static_assert(internal::collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(internal::collection<std::vector<angle::Radians>>);

  static_assert(vector_space_descriptor_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(vector_space_descriptor_tuple<std::array<Distance, 5>>);
  static_assert(not vector_space_descriptor_tuple<std::tuple<Axis, Distance, double, angle::Radians>>);
  static_assert(not vector_space_descriptor_tuple<std::vector<angle::Radians>>);
  static_assert(not vector_space_descriptor_tuple<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_vector_space_descriptor_tuple<std::tuple<Axis, Dimensions<3>, int, std::integral_constant<std::size_t, 5>>>);
  static_assert(not euclidean_vector_space_descriptor_tuple<std::tuple<Axis, Distance, angle::Radians>>);

  static_assert(vector_space_descriptor_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(vector_space_descriptor_collection<std::array<Distance, 5>>);
  static_assert(vector_space_descriptor_collection<std::vector<angle::Radians>>);
  static_assert(vector_space_descriptor_collection<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_vector_space_descriptor_collection<std::tuple<Axis, Dimensions<3>, int, std::integral_constant<std::size_t, 5>>>);
  static_assert(euclidean_vector_space_descriptor_collection<std::array<Dimensions<4>, 5>>);
  static_assert(euclidean_vector_space_descriptor_collection<std::vector<Axis>>);
  static_assert(euclidean_vector_space_descriptor_collection<std::initializer_list<int>>);
  static_assert(not euclidean_vector_space_descriptor_collection<std::array<Distance, 5>>);
  static_assert(not euclidean_vector_space_descriptor_collection<std::vector<angle::Radians>>);
  static_assert(not euclidean_vector_space_descriptor_collection<std::initializer_list<Distance>>);

  static_assert(static_vector_space_descriptor_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(not static_vector_space_descriptor_tuple<std::tuple<Axis, Distance, int, angle::Radians>>);

  static_assert(static_vector_space_descriptor_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(static_vector_space_descriptor_collection<std::array<Distance, 5>>);
  static_assert(static_vector_space_descriptor_collection<std::vector<angle::Radians>>);
  static_assert(static_vector_space_descriptor_collection<std::initializer_list<angle::Radians>>);
  static_assert(not static_vector_space_descriptor_collection<std::tuple<Axis, int, Distance, angle::Radians>>);
  static_assert(not static_vector_space_descriptor_collection<std::array<Dimensions<dynamic_size>, 5>>);
  static_assert(not static_vector_space_descriptor_collection<std::vector<int>>);
  static_assert(not static_vector_space_descriptor_collection<std::initializer_list<int>>);
}

