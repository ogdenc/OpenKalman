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


TEST(coefficients, integral)
{
  static_assert(not composite_coefficients<int>);
  static_assert(not atomic_coefficient_group<int>);
  static_assert(untyped_index_descriptor<int>);
  static_assert(not typed_index_descriptor<int>);
  static_assert(dimension_size_of_v<int> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<int> == dynamic_size);
  static_assert(get_dimension_size_of(3) == 3);
  EXPECT_EQ(get_dimension_size_of(3), 3);
  static_assert(get_euclidean_dimension_size_of(3) == 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(3), 3);
}


TEST(coefficients, Dimensions)
{
  static_assert(Dimensions<3>::value == 3);
  static_assert(Dimensions<dynamic_size>::value == dynamic_size);
  static_assert(Dimensions<3>{}() == 3);
  EXPECT_EQ(Dimensions{3}(), 3);
  static_assert(Dimensions<3>{} == 3);
  EXPECT_EQ(Dimensions{3}, 3);
  static_assert(composite_coefficients<Dimensions<3>>);
  static_assert(not composite_coefficients<Dimensions<dynamic_size>>);
  static_assert(untyped_index_descriptor<Dimensions<3>>);
  static_assert(untyped_index_descriptor<Dimensions<dynamic_size>>);
  static_assert(dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(dimension_size_of_v<Dimensions<dynamic_size>> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<Dimensions<3>> == 3);
  static_assert(euclidean_dimension_size_of_v<Dimensions<dynamic_size>> == dynamic_size);
  static_assert(get_dimension_size_of(Dimensions<3>{}) == 3);
  EXPECT_EQ(get_dimension_size_of(Dimensions{3}), 3);
  static_assert(get_euclidean_dimension_size_of(Dimensions<3>{}) == 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(Dimensions{3}), 3);
}


TEST(coefficients, Axis)
{
  static_assert(Axis::value == 1);
  static_assert(Axis{}() == 1);
  static_assert(Axis{} == 1);
  static_assert(not composite_coefficients<Axis>);
  static_assert(untyped_index_descriptor<Axis>);
  static_assert(dimension_size_of_v<Axis> == 1);
  static_assert(euclidean_dimension_size_of_v<Axis> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Axis>, Axis>);
  static_assert(get_dimension_size_of(Axis{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Axis{}) == 1);
}


TEST(coefficients, Distance)
{
  static_assert(Distance::value == 1);
  static_assert(Distance{}() == 1);
  static_assert(Distance{} == 1);
  static_assert(not composite_coefficients<Distance>);
  static_assert(typed_index_descriptor<Distance>);
  static_assert(dimension_size_of_v<Distance> == 1);
  static_assert(euclidean_dimension_size_of_v<Distance> == 1);
  static_assert(std::is_same_v<dimension_difference_of_t<Distance>, Axis>);
  static_assert(get_dimension_size_of(Distance{}) == 1);
  static_assert(get_euclidean_dimension_size_of(Distance{}) == 1);
}


TEST(coefficients, Angle)
{
  static_assert(angle::Radians::value == 1);
  static_assert(angle::Radians{}() == 1);
  static_assert(angle::Radians{} == 1);
  static_assert(not composite_coefficients<angle::Radians>);
  static_assert(typed_index_descriptor<angle::Radians>);
  static_assert(dimension_size_of_v<angle::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<angle::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<angle::Radians>, angle::Radians>);
  static_assert(get_dimension_size_of(angle::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(angle::Radians{}) == 2);
}


TEST(coefficients, Inclination)
{
  static_assert(inclination::Radians::value == 1);
  static_assert(inclination::Radians{}() == 1);
  static_assert(inclination::Radians{} == 1);
  static_assert(not composite_coefficients<inclination::Radians>);
  static_assert(typed_index_descriptor<inclination::Radians>);
  static_assert(dimension_size_of_v<inclination::Radians> == 1);
  static_assert(euclidean_dimension_size_of_v<inclination::Radians> == 2);
  static_assert(std::is_same_v<dimension_difference_of_t<inclination::Radians>, Axis>);
  static_assert(get_dimension_size_of(inclination::Radians{}) == 1);
  static_assert(get_euclidean_dimension_size_of(inclination::Radians{}) == 2);
}


TEST(coefficients, Polar)
{
  static_assert(Polar<Distance, angle::Radians>::value == 2);
  static_assert(Polar<Distance, angle::Radians>{}() == 2);
  static_assert(Polar<Distance, angle::Radians>{} == 2);
  static_assert(not composite_coefficients<Polar<Distance, angle::Radians>>);
  static_assert(typed_index_descriptor<Polar<Distance, angle::Radians>>);
  static_assert(dimension_size_of_v<Polar<Distance, angle::Radians>> == 2);
  static_assert(euclidean_dimension_size_of_v<Polar<Distance, angle::Radians>> == 3);
  static_assert(std::is_same_v<dimension_difference_of_t<Polar<Distance, angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(get_dimension_size_of(Polar<Distance, angle::Radians>{}) == 2);
  static_assert(get_euclidean_dimension_size_of(Polar<Distance, angle::Radians>{}) == 3);
}


TEST(coefficients, Spherical)
{
  static_assert(Spherical<Distance, angle::Radians, inclination::Radians>::value == 3);
  static_assert(Spherical<Distance, angle::Radians, inclination::Radians>{}() == 3);
  static_assert(Spherical<Distance, angle::Radians, inclination::Radians>{} == 3);
  static_assert(not composite_coefficients<Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(typed_index_descriptor<Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<Spherical<Distance, angle::Radians, inclination::Radians>> == 4);
  static_assert(std::is_same_v<dimension_difference_of_t<Spherical<Distance, angle::Radians, inclination::Radians>>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Spherical<Distance, angle::Radians, inclination::Radians>{}) == 4);
}


TEST(coefficients, Coefficients)
{
  static_assert(dimension_size_of_v<Coefficients<Axis, Axis>> == 2);
  static_assert(euclidean_dimension_size_of_v<Coefficients<Axis, Axis>> == 2);
  static_assert(dimension_size_of_v<Coefficients<Axis, Axis, angle::Radians>> == 3);
  static_assert(euclidean_dimension_size_of_v<Coefficients<Axis, Axis, angle::Radians>> == 4);
  static_assert(untyped_index_descriptor<Coefficients<>>);
  static_assert(untyped_index_descriptor<Coefficients<Axis, Axis, Axis>>);
  static_assert(untyped_index_descriptor<Coefficients<Coefficients<Axis>>>);
  static_assert(untyped_index_descriptor<Coefficients<Coefficients<Axis>, Coefficients<Axis>>>);
  static_assert(not untyped_index_descriptor<Coefficients<Axis, Axis, angle::Radians>>);
  static_assert(not untyped_index_descriptor<Coefficients<angle::Radians, Axis, Axis>>);
  static_assert(typed_index_descriptor<Coefficients<Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis>::Coefficient<0>, Axis>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis>::Coefficient<1>, angle::Radians>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis>::Coefficient<2>, Axis>);
  static_assert(std::is_same_v<Coefficients<Dimensions<3>, Dimensions<2>, Axis>::Coefficient<0>, Dimensions<3>>);
  static_assert(std::is_same_v<Coefficients<Dimensions<3>, Dimensions<2>, Axis>::Coefficient<1>, Dimensions<2>>);
  static_assert(std::is_same_v<Coefficients<Dimensions<3>, Dimensions<2>, Axis>::Coefficient<2>, Axis>);
  static_assert(std::is_same_v<dimension_difference_of_t<Coefficients<Distance, angle::Radians, inclination::Radians>>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(get_dimension_size_of(Coefficients<Axis, Axis, angle::Radians>{}) == 3);
  static_assert(get_euclidean_dimension_size_of(Coefficients<Axis, Axis, angle::Radians>{}) == 4);
}


TEST(coefficients, prepend_append)
{
  static_assert(std::is_same_v<Coefficients<angle::Radians, Axis>::Prepend<Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, angle::Radians>::Prepend<Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Coefficients<angle::Radians, Axis>::Append<angle::Radians>, Coefficients<angle::Radians, Axis, angle::Radians>>);
  static_assert(!std::is_same_v<Coefficients<Axis, angle::Radians>::Append<angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians>::Append<Polar<Distance, angle::Radians>>, Coefficients<Axis, angle::Radians, Polar<Distance, angle::Radians>>>);
}


TEST(coefficients, Take)
{
  // Take
  static_assert(std::is_same_v<Coefficients<>::Take<0>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<angle::Radians>::Take<1>, Coefficients<angle::Radians>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, Coefficients<Axis, angle::Radians, Axis, Axis>>);
}


TEST(coefficients, Discard)
{
  // Discard
  static_assert(std::is_same_v<Coefficients<Axis>::Discard<0>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Coefficients<angle::Radians>::Discard<1>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis, Axis, angle::Radians>::Discard<3>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis, angle::Radians, Axis>::Discard<3>, Coefficients<angle::Radians, Axis>>);
}


TEST(coefficients, replicated_fixed_index_descriptor)
{
  // Replicate
  static_assert(std::is_same_v<replicated_fixed_index_descriptor<angle::Radians, 0>, Coefficients<>>);
  static_assert(std::is_same_v<replicated_fixed_index_descriptor<angle::Radians, 1>, angle::Radians>);
  static_assert(std::is_same_v<replicated_fixed_index_descriptor<angle::Radians, 2>, Coefficients<angle::Radians, angle::Radians>>);
  static_assert(std::is_same_v<replicated_fixed_index_descriptor<Coefficients<angle::Radians, Axis>, 2>, Coefficients<Coefficients<angle::Radians, Axis>, Coefficients<angle::Radians, Axis>>>);
  static_assert(std::is_same_v<replicated_fixed_index_descriptor<Dimensions<3>, 2>, Coefficients<Dimensions<3>, Dimensions<3>>>);
}


TEST(coefficients, Concatenate)
{
  // Concatenate
  static_assert(std::is_same_v<Concatenate<Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<>, Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Axis>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Coefficients<>>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<angle::Radians>, Axis>, Coefficients<angle::Radians, Axis>>);
  static_assert(std::is_same_v<Concatenate<Axis, Coefficients<angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<>, Coefficients<angle::Radians>>, Coefficients<angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Coefficients<angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis, angle::Radians>, Coefficients<angle::Radians, Axis>>, Coefficients<Axis, angle::Radians, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis, angle::Radians>, Coefficients<angle::Radians, Axis>, Coefficients<Axis, angle::Radians>>,
    Coefficients<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Polar<Distance, angle::Radians>>, Coefficients<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<Concatenate<Polar<Distance, angle::Radians>, Coefficients<Axis>>, Coefficients<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<Concatenate<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>,
    Coefficients<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>>);
}


TEST(coefficients, reduced_typed_index_descriptor)
{
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Axis>, Axis>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Dimensions<1>>, Axis>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Dimensions<3>>, Coefficients<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Dimensions<3>>>, Coefficients<Axis, Axis, Axis>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<angle::Radians>, angle::Radians>);
  static_assert(not std::is_same_v<reduced_fixed_index_descriptor_t<angle::Degrees>, angle::Radians>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Axis>>, Axis>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Axis, angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Coefficients<Axis>, angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Dimensions<3>, angle::Radians>>, Coefficients<Axis, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<angle::Radians, Dimensions<3>>>, Coefficients<angle::Radians, Axis, Axis, Axis>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Polar<Distance, Coefficients<angle::Radians>>>, Polar<Distance, angle::Radians>>);
  static_assert(std::is_same_v<reduced_fixed_index_descriptor_t<Spherical<Distance, Coefficients<angle::Radians>, inclination::Radians>>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Axis, angle::Radians, angle::Radians>>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(not std::is_same_v<reduced_fixed_index_descriptor_t<Coefficients<Axis, angle::Radians>>, Polar<Distance, angle::Radians>>);
}


TEST(coefficients, equivalent_to)
{
  static_assert(equivalent_to<Coefficients<>, Coefficients<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(equivalent_to<Dimensions<1>, Axis>);
  static_assert(equivalent_to<Coefficients<Dimensions<1>>, Coefficients<Axis>>);
  static_assert(equivalent_to<Axis, Dimensions<1>>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, Coefficients<Axis>>);
  static_assert(equivalent_to<Coefficients<Axis>, Axis>);
  static_assert(equivalent_to<Coefficients<Axis>, Coefficients<Axis>>);
  static_assert(equivalent_to<Coefficients<Axis, angle::Radians, Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Coefficients<Dimensions<2>, angle::Radians, Dimensions<3>>, Coefficients<Axis, Axis, angle::Radians, Axis, Axis, Axis>>);
  static_assert(equivalent_to<Coefficients<Axis, angle::Radians, Axis>, Coefficients<Axis, angle::Radians, Coefficients<Coefficients<Axis>>>>);
  static_assert(equivalent_to<Coefficients<Coefficients<Axis>, angle::Radians, Coefficients<Axis>>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<Distance, Coefficients<angle::Radians>>, Polar<Distance, angle::Radians>>);
  static_assert(equivalent_to<Spherical<Distance, Coefficients<angle::Radians>, inclination::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not equivalent_to<Coefficients<Axis, angle::Radians, angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(not equivalent_to<Coefficients<Axis, angle::Radians>, Polar<Distance, angle::Radians>>);
}


TEST(coefficients, prefix_of)
{
  static_assert(prefix_of<Coefficients<>, Axis>);
  static_assert(prefix_of<Coefficients<>, Dimensions<2>>);
  static_assert(prefix_of<Coefficients<>, Coefficients<Axis>>);
  static_assert(prefix_of<Coefficients<>, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Coefficients<Axis>, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Coefficients<Axis>, Coefficients<Dimensions<2>, angle::Radians>>);
  static_assert(prefix_of<Axis, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Axis, Coefficients<Dimensions<2>, angle::Radians>>);
  static_assert(not prefix_of<angle::Radians, Coefficients<Axis, angle::Radians>>);
  static_assert(not prefix_of<Coefficients<angle::Radians>, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Coefficients<Axis, angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(prefix_of<Coefficients<Axis, angle::Radians, Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<Coefficients<Axis, angle::Radians, angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
}


TEST(coefficients, has_uniform_dimension_type)
{
  static_assert(not has_uniform_dimension_type<Coefficients<>>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Axis>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Distance>, Distance>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<angle::Radians>, angle::Radians>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<inclination::Radians>, inclination::Radians>);
  static_assert(not has_uniform_dimension_type<Polar<>>);
  static_assert(not has_uniform_dimension_type<Spherical<>>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Coefficients<Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Dimensions<1>>, Axis>);
  static_assert(not has_uniform_dimension_type<Coefficients<Axis, angle::Radians>>);
  static_assert(has_uniform_dimension_type<Coefficients<Axis, Axis>>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Coefficients<Axis, Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Coefficients<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Dimensions<2>>, Axis>);
  static_assert(std::is_same_v<uniform_dimension_type_of_t<Dimensions<3>>, Axis>);
  static_assert(not has_uniform_dimension_type<Coefficients<Polar<>, Polar<>>>);
}


TEST(coefficients, arithmetic)
{
  static_assert(Dimensions<3>{} == Dimensions<3>{});
  static_assert(Dimensions<3>{} != Dimensions<4>{});
  static_assert(Dimensions<3>{} < Dimensions<4>{});
  static_assert(Dimensions<4>{} > Dimensions<3>{});
  static_assert(Dimensions<3>{} == Coefficients<Axis, Axis, Axis>{});
  static_assert(Coefficients<Axis, angle::Radians>{} != Dimensions<5>{});
  static_assert(Dimensions<3>{} + Dimensions<4>{} == Dimensions<7>{});
  EXPECT_TRUE(Dimensions{3} + Dimensions{4} == Dimensions{7});
  static_assert(Coefficients<Axis, Axis>{} + Coefficients<Axis, Axis, Axis>{} == Dimensions<5>{});
  static_assert(Dimensions<7>{} - Dimensions<4>{} == Dimensions<3>{});
  EXPECT_TRUE(Dimensions{7} - Dimensions{4} == Dimensions{3});
}


TEST(coefficients, equal_dynamic)
{
  EXPECT_TRUE(DynamicCoefficients {Coefficients<> {}} == DynamicCoefficients {Coefficients<> {}});
  EXPECT_TRUE(DynamicCoefficients {Coefficients<> {}} == DynamicCoefficients {});
  EXPECT_TRUE(DynamicCoefficients {Axis {}} == DynamicCoefficients {Axis {}});
  EXPECT_FALSE(DynamicCoefficients {Axis {}} == DynamicCoefficients {angle::Radians {}});
  EXPECT_FALSE(DynamicCoefficients {angle::Degrees {}} == DynamicCoefficients {angle::Radians {}});
  EXPECT_FALSE(DynamicCoefficients {Axis {}} == DynamicCoefficients {Polar<> {}});
  EXPECT_TRUE(DynamicCoefficients {Axis {}} == DynamicCoefficients {Coefficients<Axis> {}});
  EXPECT_TRUE(DynamicCoefficients {Coefficients<Axis> {}} == DynamicCoefficients {Axis {}});
  EXPECT_TRUE(DynamicCoefficients {Coefficients<Axis> {}} == DynamicCoefficients {Coefficients<Axis> {}});
  EXPECT_TRUE((DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}} == DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}} == DynamicCoefficients {Axis {}, angle::Radians {}, Axis {}}));
  EXPECT_TRUE((DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}} == DynamicCoefficients {Axis {}, angle::Radians {}, Coefficients<Axis> {}}));
  EXPECT_TRUE((DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}} == DynamicCoefficients {Axis {}, angle::Radians {}, Coefficients<Coefficients<Axis>> {}}));
  EXPECT_TRUE((DynamicCoefficients {Coefficients<Coefficients<Axis>, angle::Radians, Coefficients<Axis>> {}} == DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicCoefficients {Polar<Distance, Coefficients<angle::Radians>> {}} == DynamicCoefficients {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicCoefficients {Spherical<Distance, Coefficients<angle::Radians>, inclination::Radians> {}} == DynamicCoefficients {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_FALSE((DynamicCoefficients {Coefficients<Axis, angle::Radians, angle::Radians> {}} == DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis> {}}));
  EXPECT_FALSE((DynamicCoefficients {Coefficients<Axis, angle::Radians> {}} == DynamicCoefficients {Polar<Distance, angle::Radians> {}}));
}


int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
