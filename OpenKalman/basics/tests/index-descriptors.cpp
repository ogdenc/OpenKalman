/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
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


TEST(index_descriptors, integral_constant)
{
  static_assert(dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(dimension_size_of_v<std::integral_constant<int, 3>> == 3);
  static_assert(euclidean_dimension_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(index_descriptor_components_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(get_dimension_size_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_index_descriptor_component_count_of(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(fixed_index_descriptor<std::integral_constant<std::size_t, 3>>);
  static_assert(fixed_index_descriptor<std::integral_constant<int, 3>>);
  static_assert(not composite_index_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(not composite_index_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(atomic_fixed_index_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(atomic_fixed_index_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(euclidean_index_descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(euclidean_index_descriptor<std::integral_constant<std::size_t, 2>>);
}


TEST(index_descriptors, fixed_Dimensions)
{
  static_assert(dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(euclidean_dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(index_descriptor_components_of_v<Dimensions<3>> == 3);
  static_assert(get_dimension_size_of(Dimensions<3>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Dimensions<3>{}) == 3);
  static_assert(get_index_descriptor_component_count_of(Dimensions<3>{}) == 3);
  static_assert(get_dimension_size_of(Dimensions{Axis {}}) == 1);
  static_assert(get_dimension_size_of(Dimensions{TypedIndex<Axis, Axis> {}}) == 2);
  static_assert(get_dimension_size_of(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(fixed_index_descriptor<Dimensions<3>>);
  static_assert(not composite_index_descriptor<Dimensions<1>>);
  static_assert(not composite_index_descriptor<Dimensions<2>>);
  static_assert(atomic_fixed_index_descriptor<Dimensions<1>>);
  static_assert(atomic_fixed_index_descriptor<Dimensions<2>>);
  static_assert(euclidean_index_descriptor<Dimensions<1>>);
  static_assert(euclidean_index_descriptor<Dimensions<2>>);
  static_assert(static_cast<std::integral_constant<int, 3>>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
}


TEST(index_descriptors, Axis)
{
  static_assert(dimension_size_of_v<Axis> == 1);
  static_assert(euclidean_dimension_size_of_v<Axis> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Axis>, Axis>);
  static_assert(get_dimension_size_of(Axis{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Axis{}) == 1);
  static_assert(not composite_index_descriptor<Axis>);
  static_assert(euclidean_index_descriptor<Axis>);
}


TEST(index_descriptors, Distance)
{
  static_assert(dimension_size_of_v<Distance> == 1);
  static_assert(euclidean_dimension_size_of_v<Distance> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Distance>, Axis>);
  static_assert(get_dimension_size_of(Distance{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Distance{}) == 1);
  static_assert(not composite_index_descriptor<Distance>);
  static_assert(fixed_index_descriptor<Distance>);
}


TEST(index_descriptors, Angle)
{
  static_assert(dimension_size_of_v<angle::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<angle::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<angle::Radians>, angle::Radians>);
  static_assert(get_dimension_size_of(angle::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(angle::Radians{}) == 2);
  static_assert(not composite_index_descriptor<angle::Radians>);
  static_assert(fixed_index_descriptor<angle::Radians>);
}


TEST(index_descriptors, Inclination)
{
  static_assert(dimension_size_of_v<inclination::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<inclination::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<inclination::Radians>, Axis>);
  static_assert(get_dimension_size_of(inclination::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(inclination::Radians{}) == 2);
  static_assert(not composite_index_descriptor<inclination::Radians>);
  static_assert(fixed_index_descriptor<inclination::Radians>);
}


TEST(index_descriptors, Polar)
{
  static_assert(dimension_size_of_v<Polar<Distance, angle::Radians>> == 2);
  static_assert(euclidean_dimension_size_of_v<Polar<Distance, angle::Radians>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<Polar<Distance, angle::Radians>>, TypedIndex<Axis, angle::Radians>>);
  static_assert(get_dimension_size_of(Polar<Distance, angle::Radians>{}) == 2);
  static_assert(get_euclidean_dimension_size_of(Polar<Distance, angle::Radians>{}) == 3);
  static_assert(not composite_index_descriptor<Polar<Distance, angle::Radians>>);
  static_assert(fixed_index_descriptor<Polar<Distance, angle::Radians>>);
}


TEST(index_descriptors, Spherical)
{
  static_assert(dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 4);
  static_assert(std::is_same_v<dimension_difference_of_t<Spherical<Distance, angle::Radians, inclination::Radians>>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 4);
  static_assert(not composite_index_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(fixed_index_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
}


TEST(index_descriptors, TypedIndex)
{
  static_assert(dimension_size_of_v<TypedIndex<Axis, Axis>> == 2);
  static_assert(euclidean_dimension_size_of_v<TypedIndex<Axis, Axis>> == 2);
  static_assert(index_descriptor_components_of_v<TypedIndex<Axis, Axis>> == 2);
  static_assert(dimension_size_of_v<TypedIndex<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<TypedIndex<Axis, Axis, angle::Radians>> == 4);
  static_assert(index_descriptor_components_of_v<TypedIndex<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_index_descriptor<TypedIndex<>>);
  static_assert(euclidean_index_descriptor<TypedIndex<Axis, Axis, Axis>>);
  static_assert(euclidean_index_descriptor<TypedIndex<TypedIndex<Axis>>>);
  static_assert(euclidean_index_descriptor<TypedIndex<TypedIndex<Axis>, TypedIndex<Axis>>>);
  static_assert(not euclidean_index_descriptor<TypedIndex<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_index_descriptor<TypedIndex<angle::Radians, Axis, Axis>>);
  static_assert(fixed_index_descriptor<TypedIndex<Axis, Axis, angle::Radians>>);
  static_assert(not atomic_fixed_index_descriptor<TypedIndex<Axis>>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians, Axis>::Select<0>, Axis>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians, Axis>::Select<1>, angle::Radians>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians, Axis>::Select<2>, Axis>);
  static_assert(std::is_same_v<TypedIndex<Dimensions<3>, Dimensions<2>, Axis>::Select<0>, Dimensions<3>>);
  static_assert(std::is_same_v<TypedIndex<Dimensions<3>, Dimensions<2>, Axis>::Select<1>, Dimensions<2>>);
  static_assert(std::is_same_v<TypedIndex<Dimensions<3>, Dimensions<2>, Axis>::Select<2>, Axis>);
  static_assert(std::is_same_v<dimension_difference_of_t<TypedIndex<Distance, angle::Radians, inclination::Radians>>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(TypedIndex<Axis, Axis, angle::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(TypedIndex<Axis, Axis, angle::Radians>{}) == 4);
  static_assert(get_index_descriptor_component_count_of(TypedIndex<Axis, TypedIndex<Axis, angle::Radians>, angle::Radians>{}) == 4);
}


TEST(index_descriptors, prepend_append)
{
  static_assert(std::is_same_v<TypedIndex<angle::Radians, Axis>::Prepend<Axis>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<TypedIndex<Axis, angle::Radians>::Prepend<Axis>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<TypedIndex<angle::Radians, Axis>::Append<angle::Radians>, TypedIndex<angle::Radians, Axis, angle::Radians>>);
  static_assert(!std::is_same_v<TypedIndex<Axis, angle::Radians>::Append<angle::Radians>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians>::Append<Polar<Distance, angle::Radians>>, TypedIndex<Axis, angle::Radians, Polar<Distance, angle::Radians>>>);
}


TEST(index_descriptors, Take)
{
  static_assert(std::is_same_v<TypedIndex<>::Take<0>, TypedIndex<>>);
  static_assert(std::is_same_v<TypedIndex<angle::Radians>::Take<1>, TypedIndex<angle::Radians>>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<TypedIndex<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, TypedIndex<Axis, angle::Radians, Axis, Axis>>);
}


TEST(index_descriptors, Discard)
{
  static_assert(std::is_same_v<TypedIndex<Axis>::Discard<0>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<TypedIndex<angle::Radians>::Discard<1>, TypedIndex<>>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians, Axis, Axis, angle::Radians>::Discard<3>, TypedIndex<Axis, angle::Radians>>);
  static_assert(std::is_same_v<TypedIndex<Axis, angle::Radians, Axis, angle::Radians, Axis>::Discard<3>, TypedIndex<angle::Radians, Axis>>);
}


TEST(index_descriptors, replicate_fixed_index_descriptor)
{
  static_assert(std::is_same_v<replicate_fixed_index_descriptor_t<angle::Radians, 0>, TypedIndex<>>);
  static_assert(std::is_same_v<replicate_fixed_index_descriptor_t<angle::Radians, 1>, angle::Radians>);
  static_assert(std::is_same_v<replicate_fixed_index_descriptor_t<angle::Radians, 2>, TypedIndex<angle::Radians, angle::Radians>>);
  static_assert(std::is_same_v<replicate_fixed_index_descriptor_t<TypedIndex<angle::Radians, Axis>, 2>, TypedIndex<TypedIndex<angle::Radians, Axis>, TypedIndex<angle::Radians, Axis>>>);
  static_assert(std::is_same_v<replicate_fixed_index_descriptor_t<Dimensions<3>, 2>, TypedIndex<Dimensions<3>, Dimensions<3>>>);
}


TEST(index_descriptors, concatenate_fixed_index_descriptor_t)
{
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<>>, TypedIndex<>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<>, TypedIndex<>>, TypedIndex<>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<Axis>>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<Axis>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<Axis>, TypedIndex<>>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<angle::Radians>, Axis>, TypedIndex<angle::Radians, Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<Axis, TypedIndex<angle::Radians>>, TypedIndex<Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<>, TypedIndex<angle::Radians>>, TypedIndex<angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<Axis>, TypedIndex<angle::Radians>>, TypedIndex<Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<Axis, angle::Radians>, TypedIndex<angle::Radians, Axis>>, TypedIndex<Axis, angle::Radians, angle::Radians, Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t <TypedIndex<Axis, angle::Radians>, TypedIndex<angle::Radians, Axis>, TypedIndex<Axis, angle::Radians>>,
    TypedIndex<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<TypedIndex<Axis>, Polar<Distance, angle::Radians>>, TypedIndex<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<Polar<Distance, angle::Radians>, TypedIndex<Axis>>, TypedIndex<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<concatenate_fixed_index_descriptor_t<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>,
    TypedIndex<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>>);
}


TEST(index_descriptors, canonical_fixed_index_descriptor)
{
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<>>, TypedIndex<>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<Axis>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<Dimensions<1>>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<Dimensions<3>>, TypedIndex<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Dimensions<3>>>, TypedIndex<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Dimensions<3>, Dimensions<2>>>, TypedIndex<Axis, Axis, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<angle::Radians, Dimensions<1>, Dimensions<2>>>, TypedIndex<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<angle::Radians>, TypedIndex<angle::Radians>>);
  static_assert(not std::is_same_v<canonical_fixed_index_descriptor_t<angle::Degrees>, angle::Radians>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Axis>>, TypedIndex<Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Axis, angle::Radians>>, TypedIndex<Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<TypedIndex<Axis>, angle::Radians>>, TypedIndex<Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Dimensions<3>, angle::Radians>>, TypedIndex<Axis, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<angle::Radians, Dimensions<3>>>, TypedIndex<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<Polar<Distance, angle::Radians>>, TypedIndex<Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<canonical_fixed_index_descriptor_t<Spherical<Distance, angle::Radians, inclination::Radians>>, TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>>>);
  static_assert(not std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Axis, angle::Radians, angle::Radians>>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(not std::is_same_v<canonical_fixed_index_descriptor_t<TypedIndex<Axis, angle::Radians>>, TypedIndex<Polar<Distance, angle::Radians>>>);
}


TEST(index_descriptors, equivalent_to)
{
  static_assert(equivalent_to<TypedIndex<>, TypedIndex<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(equivalent_to<Dimensions<1>, Axis>);
  static_assert(equivalent_to<TypedIndex<Dimensions<1>>, TypedIndex<Axis>>);
  static_assert(equivalent_to<Axis, Dimensions<1>>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, TypedIndex<Axis>>);
  static_assert(equivalent_to<TypedIndex<Axis>, Axis>);
  static_assert(equivalent_to<TypedIndex<Axis>, TypedIndex<Axis>>);
  static_assert(equivalent_to<TypedIndex<Axis, angle::Radians, Axis>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<TypedIndex<Dimensions<2>, angle::Radians, Dimensions<3>>, TypedIndex<Axis, Axis, angle::Radians, Axis, Axis, Axis>>);
  static_assert(equivalent_to<TypedIndex<Axis, angle::Radians, Axis>, TypedIndex<Axis, angle::Radians, TypedIndex<TypedIndex<Axis>>>>);
  static_assert(equivalent_to<TypedIndex<TypedIndex<Axis>, angle::Radians, TypedIndex<Axis>>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<Distance, angle::Radians>, Polar<Distance, angle::Radians>>);
  static_assert(equivalent_to<Spherical<Distance, angle::Radians, inclination::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not equivalent_to<TypedIndex<Axis, angle::Radians, angle::Radians>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(not equivalent_to<TypedIndex<Axis, angle::Radians>, Polar<Distance, angle::Radians>>);
}


TEST(index_descriptors, prefix_of)
{
  static_assert(prefix_of<TypedIndex<>, Axis>);
  static_assert(prefix_of<TypedIndex<>, Dimensions<2>>);
  static_assert(prefix_of<TypedIndex<>, TypedIndex<Axis>>);
  static_assert(prefix_of<TypedIndex<>, TypedIndex<Axis, angle::Radians>>);
  static_assert(prefix_of<TypedIndex<Axis>, TypedIndex<Axis, angle::Radians>>);
  static_assert(prefix_of<TypedIndex<Axis>, TypedIndex<Dimensions<2>, angle::Radians>>);
  static_assert(prefix_of<Axis, TypedIndex<Axis, angle::Radians>>);
  static_assert(prefix_of<Axis, TypedIndex<Dimensions<2>, angle::Radians>>);
  static_assert(not prefix_of<angle::Radians, TypedIndex<Axis, angle::Radians>>);
  static_assert(not prefix_of<TypedIndex<angle::Radians>, TypedIndex<Axis, angle::Radians>>);
  static_assert(prefix_of<TypedIndex<Axis, angle::Radians>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(prefix_of<TypedIndex<Axis, angle::Radians, Axis>, TypedIndex<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<TypedIndex<Axis, angle::Radians, angle::Radians>, TypedIndex<Axis, angle::Radians, Axis>>);
}


TEST(index_descriptors, head_of)
{
  static_assert(equivalent_to<head_of_t<TypedIndex<>>, TypedIndex<>>);
  static_assert(equivalent_to<head_of_t<Axis>, Axis>);
  static_assert(equivalent_to<head_of_t<TypedIndex<Axis, Distance>>, Axis>);
  static_assert(equivalent_to<head_of_t<TypedIndex<Axis, Polar<Distance, angle::Radians>>>, Axis>);
  static_assert(equivalent_to<head_of_t<TypedIndex<TypedIndex<Axis, Distance>, Distance>>, Axis>);
}


TEST(index_descriptors, tail_of)
{
  static_assert(equivalent_to<tail_of_t<TypedIndex<>>, TypedIndex<>>);
  static_assert(equivalent_to<tail_of_t<Axis>, TypedIndex<>>);
  static_assert(equivalent_to<tail_of_t<TypedIndex<Axis, Distance>>, Distance>);
  static_assert(equivalent_to<tail_of_t<TypedIndex<Axis, Distance, Axis>>, TypedIndex<Distance, Axis>>);
  static_assert(equivalent_to<tail_of_t<TypedIndex<Axis, Polar<Distance, angle::Radians>>>, Polar<Distance, angle::Radians>>);
  static_assert(equivalent_to<tail_of_t<TypedIndex<TypedIndex<Axis, Distance>, Axis>>, TypedIndex<Distance, Axis>>);
}


TEST(index_descriptors, has_uniform_dimension_type)
{
  static_assert(has_uniform_dimension_type<TypedIndex<>>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Axis>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Distance>, Distance>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<angle::Radians>, angle::Radians>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<inclination::Radians>, inclination::Radians>);
  static_assert(not has_uniform_dimension_type<Polar<>>);
  static_assert(not has_uniform_dimension_type<Spherical<>>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<TypedIndex<Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Dimensions<1>>, Axis>);
  static_assert(not has_uniform_dimension_type<TypedIndex<Axis, angle::Radians>>);
  static_assert(has_uniform_dimension_type<TypedIndex<Axis, Axis>>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<TypedIndex<Axis, Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<TypedIndex<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Dimensions<2>>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Dimensions<3>>, Axis>);
  static_assert(not has_uniform_dimension_type<TypedIndex<Polar<>, Polar<>>>);
}


TEST(index_descriptors, comparison)
{
  // Note: some tests cannot be static_assert because of a bug in GCC 10.0
  EXPECT_TRUE(Dimensions<3>{} == Dimensions<3>{});
  EXPECT_TRUE(Dimensions<3>{} <= Dimensions<3>{});
  EXPECT_TRUE(Dimensions<3>{} >= Dimensions<3>{});
  EXPECT_TRUE((Dimensions<3>{} != Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<3>{} < Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<3>{} <= Dimensions<4>{}));
  EXPECT_TRUE((Dimensions<4>{} > Dimensions<3>{}));
  EXPECT_TRUE((Dimensions<4>{} >= Dimensions<3>{}));

  EXPECT_TRUE((Dimensions<3>{} == TypedIndex<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} <= TypedIndex<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} <= TypedIndex<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<3>{} < TypedIndex<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} >= TypedIndex<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} >= TypedIndex<Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions<4>{} > TypedIndex<Axis, Axis, Axis>{}));

  EXPECT_TRUE(TypedIndex<> {} == TypedIndex<> {});
  static_assert(TypedIndex<Axis, angle::Radians>{} == TypedIndex<Axis, angle::Radians>{});
  static_assert(TypedIndex<Axis, angle::Radians>{} <= TypedIndex<Axis, angle::Radians>{});
  static_assert(TypedIndex<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{} == TypedIndex<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{});
  static_assert(TypedIndex<Axis, Dimensions<3>, angle::Radians, Axis, Dimensions<2>>{} < TypedIndex<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{});
  static_assert(TypedIndex<Axis, angle::Radians>{} < TypedIndex<Axis, angle::Radians, Axis>{});
  static_assert(TypedIndex<Axis, angle::Radians>{} >= TypedIndex<Axis, angle::Radians>{});
  static_assert(TypedIndex<Axis, angle::Radians, Axis>{} > TypedIndex<Axis, angle::Radians>{});

  static_assert(angle::Radians{} == angle::Radians{});
  static_assert(inclination::Radians{} == inclination::Radians{});
  static_assert(angle::Radians{} != inclination::Radians{});
  static_assert(not (angle::Radians{} < inclination::Radians{}));
  static_assert((Polar<Distance, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions<5>{}));
  static_assert((Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} < Dimensions<5>{}));
}


TEST(index_descriptors, fixed_arithmetic)
{
  // Note: some tests cannot be static_assert because of a bug in GCC 10.0
  EXPECT_TRUE(Dimensions<3>{} + Dimensions<4>{} == Dimensions<7>{});
  EXPECT_TRUE((TypedIndex<Axis, Axis>{} + TypedIndex<Axis, Axis, Axis>{} == Dimensions<5>{}));
  EXPECT_TRUE(Dimensions<7>{} - Dimensions<4>{} == Dimensions<3>{});
  static_assert(Polar<Distance, angle::Radians>{} + Dimensions<2>{} == TypedIndex<Polar<Distance, angle::Radians>, Axis, Axis>{});
}
