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

#include "interfaces/eigen3/tests/eigen3.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;
using std::numbers::pi;


TEST(index_descriptors, integral)
{
  static_assert(index_descriptor<int>);
  static_assert(dynamic_index_descriptor<int>);
  static_assert(not fixed_index_descriptor<int>);
  static_assert(untyped_index_descriptor<int>);
  static_assert(not composite_index_descriptor<int>);
  static_assert(not atomic_fixed_index_descriptor<int>);
  static_assert(untyped_index_descriptor<int>);
  static_assert(not typed_index_descriptor<int>);
  static_assert(dimension_size_of_v<int> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<int> == dynamic_size);
  static_assert(get_dimension_size_of(3) == 3);
  EXPECT_EQ(get_dimension_size_of(3), 3);
  static_assert(get_euclidean_dimension_size_of(3) == 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(3), 3);
}


TEST(index_descriptors, dynamic_Dimensions)
{
  using D = Dimensions<dynamic_size>;

  static_assert(index_descriptor<D>);
  static_assert(dynamic_index_descriptor<D>);
  static_assert(not fixed_index_descriptor<D>);
  static_assert(untyped_index_descriptor<D>);
  static_assert(not composite_index_descriptor<D>);
  static_assert(untyped_index_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
  EXPECT_EQ(get_dimension_size_of(Dimensions{3}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(Dimensions{3}), 3);
}


TEST(index_descriptors, DynamicCoefficients_traits)
{
  using D = DynamicCoefficients<double>;

  static_assert(index_descriptor<D>);
  static_assert(dynamic_index_descriptor<D>);
  static_assert(not fixed_index_descriptor<D>);
  static_assert(not untyped_index_descriptor<D>);
  //static_assert(not index_descriptor_group<D>);
  static_assert(not typed_index_descriptor<D>);
  static_assert(composite_index_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
}


TEST(index_descriptors, DynamicCoefficients_construct)
{
  using D = DynamicCoefficients<double>;

  EXPECT_EQ(get_dimension_size_of(D {Axis{}}), 1);
  EXPECT_EQ(get_dimension_size_of(D {angle::Degrees{}}), 1);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {angle::Degrees{}}), 2);
  EXPECT_EQ(get_dimension_size_of(D {Dimensions<5>{}}), 5);
  EXPECT_EQ(get_dimension_size_of(D {Dimensions{5}}), 5);
  EXPECT_EQ(get_dimension_size_of(D {Polar<Distance, angle::PositiveRadians>{}}), 2);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Polar<angle::PositiveDegrees, Distance>{}}), 3);
  EXPECT_EQ(get_dimension_size_of(D {Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4);

  EXPECT_EQ(get_dimension_size_of(D {Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Axis{}, Dimensions{5}, angle::Degrees{}}), 8);
  EXPECT_EQ(get_dimension_size_of(D {Coefficients<Axis, inclination::Radians>{}, angle::Degrees{}, Dimensions{5}}), 8);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Dimensions{5}, Coefficients<Axis, inclination::Radians>{}, angle::Degrees{}}), 10);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Dimensions{5}, Coefficients<Axis, inclination::Radians>{}, D {Coefficients<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
}


TEST(index_descriptors, DynamicCoefficients_extend)
{
  using D = DynamicCoefficients<double>;

  D d;
  EXPECT_EQ(get_dimension_size_of(d), 0); EXPECT_EQ(get_euclidean_dimension_size_of(d), 0); EXPECT_EQ(get_index_descriptor_component_count_of(d), 0);
  d.extend(Axis{});
  EXPECT_EQ(get_dimension_size_of(d), 1); EXPECT_EQ(get_euclidean_dimension_size_of(d), 1); EXPECT_EQ(get_index_descriptor_component_count_of(d), 1);
  d.extend(Dimensions{5}, Dimensions<5>{}, angle::Radians{}, Coefficients<Axis, inclination::Radians>{}, Polar<angle::Degrees, Distance>{});
  EXPECT_EQ(get_dimension_size_of(d), 16); EXPECT_EQ(get_euclidean_dimension_size_of(d), 19); EXPECT_EQ(get_index_descriptor_component_count_of(d), 15);
}


TEST(index_descriptors, toEuclidean_dynamic)
{
  using M = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using DM = DynamicCoefficients<double>;

  M m2(2, 2); m2 << 1, 2, 3, 4;

  EXPECT_NEAR((DM{Axis{}, Axis{}}.to_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.to_euclidean_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.to_euclidean_element(m2, 1, 0)), 3., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.to_euclidean_element(m2, 1, 1)), 4., 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis, Axis>{}}.to_euclidean_element(m2, 0, 0)), 1., 1e-6);

  EXPECT_NEAR((DM{Dimensions<2>{}}.to_euclidean_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis>{}, Axis{}}.to_euclidean_element(m2, 1, 0)), 3., 1e-6);
  EXPECT_NEAR((DM{Distance{}, Axis{}}.to_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Distance{}}.to_euclidean_element(m2, 1, 1)), 4., 1e-6);

  m2 << 1, 2, 60, 30;

  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.to_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.to_euclidean_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.to_euclidean_element(m2, 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.to_euclidean_element(m2, 1, 1)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.to_euclidean_element(m2, 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis, angle::Degrees>{}}.to_euclidean_element(m2, 2, 1)), 0.5, 1e-6);

  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.to_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.to_euclidean_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.to_euclidean_element(m2, 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.to_euclidean_element(m2, 1, 1)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.to_euclidean_element(m2, 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis, inclination::Degrees>{}}.to_euclidean_element(m2, 2, 1)), 0.5, 1e-6);

  m2 << pi/6, -pi/4, 2, -3;

  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.to_euclidean_element(m2, 0, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.to_euclidean_element(m2, 0, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.to_euclidean_element(m2, 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.to_euclidean_element(m2, 1, 1)), -std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.to_euclidean_element(m2, 2, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Coefficients<angle::Radians, Axis>{}}.to_euclidean_element(m2, 2, 1)), -3., 1e-6);

  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.to_euclidean_element(m2, 0, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.to_euclidean_element(m2, 0, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.to_euclidean_element(m2, 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.to_euclidean_element(m2, 1, 1)), -std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.to_euclidean_element(m2, 2, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Coefficients<inclination::Radians, Axis>{}}.to_euclidean_element(m2, 2, 1)), -3., 1e-6);

  M mp(3, 2); mp << 2, -3., pi/3, pi/4, 4, 5;

  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 0, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 0, 1)), -3., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 1, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 2, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 3, 0)), 4., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.to_euclidean_element(mp, 3, 1)), 5., 1e-6);

  mp << 4, 5, pi/3, pi/4, 2, -3.;

  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 0, 0)), 4., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 0, 1)), 5., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 1, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 2, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 3, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.to_euclidean_element(mp, 3, 1)), -3., 1e-6);

  M ms(4, 2); ms << 5, 7, 2, -3., pi/6, pi/4, pi/3, pi/4;

  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 0, 0)), 5., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 0, 1)), 7., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 1, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 1, 1)), -3., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 2, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 2, 1)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 3, 0)), 0.25, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 3, 1)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 4, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(ms, 4, 1)), std::sqrt(2)/2, 1e-6);

  ms << pi/3, pi/4, pi/6, pi/4, 2, -3, 5, 7;

  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 0, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 0, 1)), -3., 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 1, 1)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 2, 1)), 0.5, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 3, 1)), std::sqrt(2)/2, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 4, 0)), 5., 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.to_euclidean_element(ms, 4, 1)), 7., 1e-6);
}


TEST(index_descriptors, fromEuclidean_dynamic)
{
  using M = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using DM = DynamicCoefficients<double>;

  M m2(2, 2); m2 << 1, 2, 3, 4;

  EXPECT_NEAR((DM{Axis{}, Axis{}}.from_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.from_euclidean_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.from_euclidean_element(m2, 1, 0)), 3., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.from_euclidean_element(m2, 1, 1)), 4., 1e-6);

  EXPECT_NEAR((DM{Coefficients<Axis, Axis>{}}.from_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Dimensions<2>{}}.from_euclidean_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis>{}, Axis{}}.from_euclidean_element(m2, 1, 0)), 3., 1e-6);
  EXPECT_NEAR((DM{Distance{}, Axis{}}.from_euclidean_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Distance{}}.from_euclidean_element(m2, 1, 1)), 4., 1e-6);

  M m3(3, 2); m3 << 1, 2, 0.5, std::sqrt(3)/2, std::sqrt(3)/2, 0.5;

  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.from_euclidean_element(m3, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.from_euclidean_element(m3, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.from_euclidean_element(m3, 1, 0)), 60, 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis, angle::Degrees>{}}.from_euclidean_element(m3, 1, 1)), 30, 1e-6);

  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.from_euclidean_element(m3, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.from_euclidean_element(m3, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.from_euclidean_element(m3, 1, 0)), 60, 1e-6);
  EXPECT_NEAR((DM{Coefficients<Axis, inclination::Degrees>{}}.from_euclidean_element(m3, 1, 1)), 30, 1e-6);

  m3 << std::sqrt(3)/2, std::sqrt(2)/2, 0.5, -std::sqrt(2)/2, 2, -3;

  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.from_euclidean_element(m3, 0, 0)), pi/6, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.from_euclidean_element(m3, 0, 1)), -pi/4, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.from_euclidean_element(m3, 1, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Coefficients<angle::Radians, Axis>{}}.from_euclidean_element(m3, 1, 1)), -3., 1e-6);

  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.from_euclidean_element(m3, 0, 0)), pi/6, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.from_euclidean_element(m3, 0, 1)), -pi/4, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.from_euclidean_element(m3, 1, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Coefficients<inclination::Radians, Axis>{}}.from_euclidean_element(m3, 1, 1)), -3., 1e-6);

  M mp(4, 2); mp << 2, -3., 0.5, std::sqrt(2)/2, std::sqrt(3)/2, std::sqrt(2)/2, 4, 5;

  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.from_euclidean_element(mp, 0, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.from_euclidean_element(mp, 0, 1)), 3., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.from_euclidean_element(mp, 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.from_euclidean_element(mp, 1, 1)), -3*pi/4, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.from_euclidean_element(mp, 2, 0)), 4., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.from_euclidean_element(mp, 2, 1)), 5., 1e-6);

  mp << 4, 5, 0.5, std::sqrt(2)/2, std::sqrt(3)/2, std::sqrt(2)/2, 2, -3.;

  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.from_euclidean_element(mp, 0, 0)), 4., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.from_euclidean_element(mp, 0, 1)), 5., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.from_euclidean_element(mp, 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.from_euclidean_element(mp, 1, 1)), -3*pi/4, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.from_euclidean_element(mp, 2, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.from_euclidean_element(mp, 2, 1)), 3., 1e-6);

  M ms(5, 2); ms << 5, 7, 2, -3., std::sqrt(3)/4, 0.5, 0.25, 0.5, std::sqrt(3)/2, std::sqrt(2)/2;

  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 0, 0)), 5., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 0, 1)), 7., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 1, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 1, 1)), 3., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 2, 0)), pi/6, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 2, 1)), -3*pi/4, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 3, 0)), pi/3, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(ms, 3, 1)), -pi/4, 1e-6);

  ms << 2, -3., std::sqrt(3)/4, 0.5, 0.25, 0.5, std::sqrt(3)/2, std::sqrt(2)/2, 5, 7;

  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 0, 1)), -pi/4, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 1, 1)), -3*pi/4, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 2, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 2, 1)), 3., 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 3, 0)), 5., 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.from_euclidean_element(ms, 3, 1)), 7., 1e-6);
}


TEST(index_descriptors, wrap_get_element_dynamic)
{
  using M = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using DM = DynamicCoefficients<double>;

  M m2(2, 2); m2 << 1, -2, -3, 4;

  EXPECT_NEAR((DM{Axis{}, Axis{}}.wrap_get_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.wrap_get_element(m2, 0, 1)), -2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.wrap_get_element(m2, 1, 0)), -3., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Axis{}}.wrap_get_element(m2, 1, 1)), 4., 1e-6);

  EXPECT_NEAR((DM{Distance{}, Distance{}}.wrap_get_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Distance{}, Distance{}}.wrap_get_element(m2, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Distance{}}.wrap_get_element(m2, 1, 0)), 3., 1e-6);
  EXPECT_NEAR((DM{Distance{}, Axis{}}.wrap_get_element(m2, 1, 1)), 4., 1e-6);

  m2 << 1, -2, 390, -100;

  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.wrap_get_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.wrap_get_element(m2, 0, 1)), -2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::Degrees{}}.wrap_get_element(m2, 1, 0)), 30, 1e-6);
  EXPECT_NEAR((DM{Axis{}, angle::PositiveDegrees{}}.wrap_get_element(m2, 1, 1)), 260, 1e-6);

  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.wrap_get_element(m2, 0, 0)), 1., 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.wrap_get_element(m2, 0, 1)), -2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.wrap_get_element(m2, 1, 0)), 30, 1e-6);
  EXPECT_NEAR((DM{Axis{}, inclination::Degrees{}}.wrap_get_element(m2, 1, 1)), -80, 1e-6);

  m2 << 13*pi/6, -3*pi/4, 2, -3;

  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.wrap_get_element(m2, 0, 0)), pi/6, 1e-6);
  EXPECT_NEAR((DM{angle::PositiveRadians{}, Axis{}}.wrap_get_element(m2, 0, 1)), 5*pi/4, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.wrap_get_element(m2, 1, 0)), 2, 1e-6);
  EXPECT_NEAR((DM{angle::Radians{}, Axis{}}.wrap_get_element(m2, 1, 1)), -3, 1e-6);

  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.wrap_get_element(m2, 0, 0)), pi/6, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.wrap_get_element(m2, 0, 1)), -pi/4, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.wrap_get_element(m2, 1, 0)), 2, 1e-6);
  EXPECT_NEAR((DM{inclination::Radians{}, Axis{}}.wrap_get_element(m2, 1, 1)), -3, 1e-6);

  M mp(3, 2); mp <<
    2, -2.,
    1.1*pi, 1.1*pi,
    4, 5;

  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_get_element(mp, 0, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_get_element(mp, 0, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_get_element(mp, 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_get_element(mp, 1, 1)), 0.1*pi, 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_get_element(mp, 2, 0)), 4., 1e-6);
  EXPECT_NEAR((DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_get_element(mp, 2, 1)), 5., 1e-6);

  mp <<
    4, 5,
    1.1*pi, 1.1*pi,
    2, -2;

  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_get_element(mp, 0, 0)), 4., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_get_element(mp, 0, 1)), 5., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_get_element(mp, 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_get_element(mp, 1, 1)), 0.1*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_get_element(mp, 2, 0)), 2, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_get_element(mp, 2, 1)), 2, 1e-6);

  M ms(4, 4); ms <<
    5, 7, -9, 11,
    2, -2, 2, -2,
    1.1*pi, 1.1*pi, 0.5*pi, 0.5*pi,
    0.6*pi, 0.6*pi, 0.6*pi, 0.6*pi;

  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 0, 2)), -9., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 1, 0)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 1, 1)), 2., 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 2, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 2, 1)), -0.9*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 2, 2)), -0.5*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 2, 3)), 0.5*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 3, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 3, 1)), -0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 3, 2)), 0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(ms, 3, 3)), -0.4*pi, 1e-6);

  ms <<
    0.6*pi, 0.6*pi, 0.6*pi, 0.6*pi,
    1.1*pi, 1.1*pi, 0.5*pi, 0.5*pi,
    2, -2, 2, -2,
    5, -7, 9, 11;

  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 0, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 0, 1)), -0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 0, 2)), 0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 0, 3)), -0.4*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 1, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 1, 1)), -0.9*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 1, 2)), -0.5*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 1, 3)), 0.5*pi, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 2, 0)), 2, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 2, 1)), 2, 1e-6);
  EXPECT_NEAR((DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_get_element(ms, 3, 1)), -7, 1e-6);
}


TEST(index_descriptors, wrap_set_element_dynamic)
{
  using M = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using DM = DynamicCoefficients<double>;

  M m2(2, 2); m2 << 1, -2, -3, 4;

  DM{Axis{}, Axis{}}.wrap_set_element(m2, -5, 0, 0);
  DM{Axis{}, Axis{}}.wrap_set_element(m2, 6, 0, 1);
  DM{Axis{}, Axis{}}.wrap_set_element(m2, 7, 1, 0);
  DM{Axis{}, Axis{}}.wrap_set_element(m2, -8, 1, 1);
  EXPECT_TRUE(is_near(m2, make_eigen_matrix<double, 2, 2>(-5, 6, 7, -8)));

  DM{Distance{}, Distance{}}.wrap_set_element(m2, 1, 0, 0);
  DM{Distance{}, Distance{}}.wrap_set_element(m2, -2, 0, 1);
  DM{Distance{}, Distance{}}.wrap_set_element(m2, -3, 1, 0);
  DM{Distance{}, Distance{}}.wrap_set_element(m2, 4, 1, 1);
  EXPECT_TRUE(is_near(m2, make_eigen_matrix<double, 2, 2>(1, 2, 3, 4)));

  DM{Axis{}, angle::Degrees{}}.wrap_set_element(m2, 1, 0, 0);
  DM{Axis{}, angle::Degrees{}}.wrap_set_element(m2, -2, 0, 1);
  DM{Axis{}, angle::Degrees{}}.wrap_set_element(m2, 390, 1, 0);
  DM{Axis{}, angle::PositiveDegrees{}}.wrap_set_element(m2, -100, 1, 1);
  EXPECT_TRUE(is_near(m2, make_eigen_matrix<double, 2, 2>(1, -2, 30, 260)));

  DM{Axis{}, inclination::Degrees{}}.wrap_set_element(m2, 460, 1, 0);
  DM{Axis{}, inclination::Degrees{}}.wrap_set_element(m2, -100, 1, 1);
  EXPECT_TRUE(is_near(m2, make_eigen_matrix<double, 2, 2>(1, -2, 80, -80)));

  DM{angle::Radians{}, Axis{}}.wrap_set_element(m2, 13*pi/6, 0, 0);
  DM{angle::PositiveRadians{}, Axis{}}.wrap_set_element(m2, -3*pi/4, 0, 1);
  DM{angle::Radians{}, Axis{}}.wrap_set_element(m2, 2, 1, 0); //
  DM{angle::Radians{}, Axis{}}.wrap_set_element(m2, -3, 1, 1);
  EXPECT_TRUE(is_near(m2, make_eigen_matrix<double, 2, 2>(pi/6, 5*pi/4, 2, -3)));

  DM{inclination::Radians{}, Axis{}}.wrap_set_element(m2, 13*pi/6, 0, 0);
  DM{inclination::Radians{}, Axis{}}.wrap_set_element(m2, -3*pi/4, 0, 1);
  DM{inclination::Radians{}, Axis{}}.wrap_set_element(m2, 2, 1, 0);
  DM{inclination::Radians{}, Axis{}}.wrap_set_element(m2, -3, 1, 1);
  EXPECT_TRUE(is_near(m2, make_eigen_matrix<double, 2, 2>(pi/6, -pi/4, 2, -3)));

  M mp(3, 2); mp << 1, 1, 0.1*pi, 0.1*pi, 1, 1;

  DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_set_element(mp, 2, 0, 0); EXPECT_NEAR(mp(0, 0), 2, 1e-6);
  DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_set_element(mp, -2, 0, 1); EXPECT_NEAR(mp(0, 1), 2, 1e-6); EXPECT_NEAR(mp(1, 1), -0.9*pi, 1e-6);
  DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_set_element(mp, 1.1*pi, 1, 0); EXPECT_NEAR(mp(0, 0), 2, 1e-6); EXPECT_NEAR(mp(1, 0), -0.9*pi, 1e-6);
  DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_set_element(mp, -2.9*pi, 1, 1); EXPECT_NEAR(mp(0, 1), 2, 1e-6); EXPECT_NEAR(mp(1, 1), -0.9*pi, 1e-6);
  DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_set_element(mp, 4, 2, 0); EXPECT_NEAR(mp(2, 0), 4, 1e-6);
  DM{Polar<Distance, angle::Radians>{}, Axis{}}.wrap_set_element(mp, 5, 2, 1); EXPECT_NEAR(mp(2, 1), 5, 1e-6);
  EXPECT_TRUE(is_near(mp, make_eigen_matrix<double, 3, 2>(2, 2, -0.9*pi, -0.9*pi, 4, 5)));

  mp << 1, 1, 0.1*pi, 0.1*pi, 1, 1;

  DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_set_element(mp, 2, 2, 0); EXPECT_NEAR(mp(2, 0), 2, 1e-6);
  DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_set_element(mp, -2, 2, 1); EXPECT_NEAR(mp(2, 1), 2, 1e-6); EXPECT_NEAR(mp(1, 1), -0.9*pi, 1e-6);
  DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_set_element(mp, 1.1*pi, 1, 0); EXPECT_NEAR(mp(2, 0), 2, 1e-6); EXPECT_NEAR(mp(1, 0), -0.9*pi, 1e-6);
  DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_set_element(mp, -2.9*pi, 1, 1); EXPECT_NEAR(mp(2, 1), 2, 1e-6); EXPECT_NEAR(mp(1, 1), -0.9*pi, 1e-6);
  DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_set_element(mp, 4, 0, 0); EXPECT_NEAR(mp(0, 0), 4, 1e-6);
  DM{Axis{}, Polar<angle::Radians, Distance>{}}.wrap_set_element(mp, 5, 0, 1); EXPECT_NEAR(mp(0, 1), 5, 1e-6);
  EXPECT_TRUE(is_near(mp, make_eigen_matrix<double, 3, 2>(4, 5, -0.9*pi, -0.9*pi, 2, 2)));


  M ms(4, 2); ms << 0, 1, 0, 1, 0, 0.1*pi, 0, 0.1*pi;

  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, -3.1, 0, 1); EXPECT_NEAR(ms(0, 1), -3.1, 1e-6);
  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, -2, 1, 1); EXPECT_NEAR(ms(1, 1), 2, 1e-6); EXPECT_NEAR(ms(2, 1), -0.9*pi, 1e-6);
  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, 1.1*pi, 2, 1); EXPECT_NEAR(ms(2, 1), -0.9*pi, 1e-6);
  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, 0.6*pi, 3, 1); EXPECT_NEAR(ms(2, 1), 0.1*pi, 1e-6); EXPECT_NEAR(ms(3, 1), 0.4*pi, 1e-6);
  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, -3, 1, 1); EXPECT_NEAR(ms(1, 1), 3, 1e-6); EXPECT_NEAR(ms(2, 1), -0.9*pi, 1e-6); EXPECT_NEAR(ms(3, 1), -0.4*pi, 1e-6);
  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, -0.6*pi, 3, 1); EXPECT_NEAR(ms(1, 1), 3, 1e-6); EXPECT_NEAR(ms(2, 1), 0.1*pi, 1e-6); EXPECT_NEAR(ms(3, 1), -0.4*pi, 1e-6);
  DM{Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_set_element(ms, -2.6*pi, 3, 1); EXPECT_NEAR(ms(1, 1), 3, 1e-6); EXPECT_NEAR(ms(2, 1), -0.9*pi, 1e-6); EXPECT_NEAR(ms(3, 1), -0.4*pi, 1e-6);
  EXPECT_TRUE(is_near(ms, make_eigen_matrix<double, 4, 2>(0, -3.1, 0, 3, 0, -0.9*pi, 0, -0.4*pi)));

  ms << 0, 0.1*pi, 0, 0.1*pi, 0, 1, 0, 1;

  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, -3.1, 3, 1); EXPECT_NEAR(ms(3, 1), -3.1, 1e-6);
  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, -2, 2, 1); EXPECT_NEAR(ms(2, 1), 2, 1e-6); EXPECT_NEAR(ms(1, 1), -0.9*pi, 1e-6);
  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, 1.1*pi, 1, 1); EXPECT_NEAR(ms(1, 1), -0.9*pi, 1e-6);
  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, 0.6*pi, 0, 1); EXPECT_NEAR(ms(1, 1), 0.1*pi, 1e-6); EXPECT_NEAR(ms(0, 1), 0.4*pi, 1e-6);
  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, -3, 2, 1); EXPECT_NEAR(ms(2, 1), 3, 1e-6); EXPECT_NEAR(ms(1, 1), -0.9*pi, 1e-6); EXPECT_NEAR(ms(0, 1), -0.4*pi, 1e-6);
  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, -0.6*pi, 0, 1); EXPECT_NEAR(ms(2, 1), 3, 1e-6); EXPECT_NEAR(ms(1, 1), 0.1*pi, 1e-6); EXPECT_NEAR(ms(0, 1), -0.4*pi, 1e-6);
  DM{Spherical<inclination::Radians, angle::Radians, Distance>{}, Axis{}}.wrap_set_element(ms, -2.6*pi, 0, 1); EXPECT_NEAR(ms(2, 1), 3, 1e-6); EXPECT_NEAR(ms(1, 1), -0.9*pi, 1e-6); EXPECT_NEAR(ms(0, 1), -0.4*pi, 1e-6);
  EXPECT_TRUE(is_near(ms, make_eigen_matrix<double, 4, 2>(0, -0.4*pi, 0, -0.9*pi, 0, 3, 0, -3.1)));
}


TEST(index_descriptors, dynamic_comparison)
{
  static_assert(Dimensions{3} == Dimensions{3});
  static_assert(Dimensions{3} == Dimensions{3});
  static_assert(Dimensions{3} <= Dimensions{3});
  static_assert(Dimensions{3} >= Dimensions{3});
  static_assert(Dimensions{3} != Dimensions{4});
  static_assert(Dimensions{3} < Dimensions{4});
  static_assert(Dimensions{3} <= Dimensions{4});
  static_assert(Dimensions{4} > Dimensions{3});
  static_assert(Dimensions{4} >= Dimensions{3});

  static_assert(Dimensions{3} == Dimensions<3>{});
  static_assert(Dimensions{3} <= Dimensions<3>{});
  static_assert(Dimensions{3} >= Dimensions<3>{});
  static_assert(Dimensions{3} != Dimensions<4>{});
  static_assert(Dimensions{3} < Dimensions<4>{});
  static_assert(Dimensions{3} <= Dimensions<4>{});
  static_assert(Dimensions{4} > Dimensions<3>{});
  static_assert(Dimensions{4} >= Dimensions<3>{});

  static_assert(Dimensions{3} == Coefficients<Axis, Axis, Axis>{});
  static_assert(Dimensions{3} <= Coefficients<Axis, Axis, Axis>{});
  static_assert(Dimensions{3} < Coefficients<Axis, Axis, Axis, Axis>{});
  static_assert(Dimensions{3} <= Coefficients<Axis, Axis, Axis, Axis>{});

  EXPECT_TRUE((Polar<Distance, angle::Radians>{} != Dimensions{5}));
  EXPECT_TRUE((Polar<Distance, angle::Radians>{} < Dimensions{5}));
  EXPECT_TRUE((Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions{5}));
  EXPECT_TRUE((Spherical<Distance, inclination::Radians, angle::Radians>{} < Dimensions{5}));

  using M = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using D = DynamicCoefficients<double>;

  EXPECT_TRUE((D {Coefficients<> {}} == D {Coefficients<> {}}));
  EXPECT_TRUE((D {Coefficients<> {}} == D {}));
  EXPECT_TRUE((D {Axis {}} == D {Axis {}}));
  EXPECT_TRUE((D {Axis {}} <= D {Axis {}}));
  EXPECT_TRUE((D {Axis {}} != D {angle::Radians {}}));
  EXPECT_TRUE((D {angle::Degrees {}} != D {angle::Radians {}}));
  EXPECT_TRUE((D {Axis {}} != D {Polar<> {}}));
  EXPECT_TRUE((D {Axis {}} == D {Coefficients<Axis> {}}));
  EXPECT_TRUE((D {Axis {}} < D {Coefficients<Axis, Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis> {}} == D {Axis {}}));
  EXPECT_TRUE((D {Coefficients<Axis, Axis> {}} > D {Axis {}}));
  EXPECT_TRUE((D {Coefficients<Axis> {}} == D {Coefficients<Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, Axis> {}} >= D {Coefficients<Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians, Axis> {}} == D {Coefficients<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians, Axis> {}} < D {Coefficients<Axis, Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians, Axis> {}} == D {Axis {}, angle::Radians {}, Axis {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians, Axis> {}} == D {Axis {}, angle::Radians {}, Coefficients<Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians, Axis> {}} == D {Axis {}, angle::Radians {}, Coefficients<Coefficients<Axis>> {}}));
  EXPECT_TRUE((D {Coefficients<Coefficients<Axis>, angle::Radians, Coefficients<Axis>> {}} == D {Coefficients<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {Polar<Distance, angle::Radians> {}} == D {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((D {Polar<Distance, angle::Radians> {}, Axis{}} > D {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((D {Spherical<Distance, angle::Radians, inclination::Radians> {}} == D {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians, angle::Radians> {}} != D {Coefficients<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians> {}} != D {Polar<Distance, angle::Radians> {}}));

  EXPECT_TRUE((D {Axis {}} == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == D {Axis {}}));
  EXPECT_TRUE((D {Coefficients<Axis> {}} == Dimensions<1>{}));
  EXPECT_TRUE((D {Coefficients<Axis, Axis, Axis> {}} == Dimensions<3>{}));
  EXPECT_TRUE((D {Coefficients<Axis, Axis> {}} < Dimensions<3>{}));
  EXPECT_TRUE((D {Dimensions<4>{}} > Coefficients<Axis, Axis> {}));
  EXPECT_TRUE((D {Dimensions<4>{}} != Coefficients<Axis, Axis> {}));
  EXPECT_TRUE((D {angle::Degrees{}} < Dimensions<3>{}));
  EXPECT_TRUE((D {Dimensions<3>{}} > angle::Degrees{}));

  EXPECT_TRUE((D {Axis {}} == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == D {Axis {}}));
  EXPECT_TRUE((D {Coefficients<Axis> {}} == Dimensions{1}));
  EXPECT_TRUE((D {Coefficients<Axis, Axis, Axis> {}} == Dimensions{3}));
  EXPECT_TRUE((D {Coefficients<Axis, Axis> {}} < Dimensions{3}));
  EXPECT_TRUE((D {Dimensions{4}} > Coefficients<Axis, Axis> {}));
  EXPECT_TRUE((Dimensions{4} != D {Coefficients<Axis, Axis> {}}));
  EXPECT_TRUE((D {angle::Degrees{}} < Dimensions{3}));

  EXPECT_TRUE((Coefficients<> {} == D {Coefficients<> {}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians>{}} == Coefficients<Axis, angle::Radians>{}));
  EXPECT_TRUE((Coefficients<Axis, angle::Radians>{} <= D {Coefficients<Axis, angle::Radians>{}}));
  EXPECT_TRUE((Coefficients<Axis, angle::Radians>{} <= D {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians>{}} >= Coefficients<Axis, angle::Radians>{}));
  EXPECT_TRUE((Coefficients<Axis, angle::Radians>{} < D {Coefficients<Axis, angle::Radians, Axis>{}}));
  EXPECT_TRUE((D {Coefficients<Axis, angle::Radians>{}} <= Coefficients<Axis, angle::Radians, Axis>{}));
  EXPECT_TRUE((angle::Radians{} == D {angle::Radians{}}));
  EXPECT_TRUE((D {inclination::Radians{}} == inclination::Radians{}));
  EXPECT_TRUE((angle::Radians{} != D {inclination::Radians{}}));
  EXPECT_FALSE((angle::Radians{} < inclination::Radians{}));
  EXPECT_TRUE((D {Polar<Distance, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_TRUE((Polar<Distance, angle::Radians>{} < D {Dimensions<5>{}}));
  EXPECT_TRUE((D {Spherical<Distance, inclination::Radians, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_TRUE((Spherical<Distance, inclination::Radians, angle::Radians>{} < D {Dimensions<5>{}}));
}


TEST(index_descriptors, dynamic_arithmetic)
{
  EXPECT_TRUE(Dimensions{3} + Dimensions{4} == Dimensions{7});
  EXPECT_TRUE(Dimensions{7} - Dimensions{4} == Dimensions{3});
}
