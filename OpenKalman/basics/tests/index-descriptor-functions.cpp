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
 * \brief Tests for coefficient functions
 */

#include <gtest/gtest.h>
#include "basics/basics.hpp"

using namespace OpenKalman;

using numbers::pi;

template<typename S = void, typename...Ss>
inline auto g(Ss...ss)
{
  using Scalar = std::conditional_t<std::is_void_v<S>, std::common_type_t<Ss...>, S>;
  return std::function {[=](std::size_t i) -> Scalar {
    auto arr = std::array<Scalar, sizeof...(Ss)> {static_cast<Scalar>(ss)...};
    return arr[i];
  }};
}


TEST(index_descriptors, toEuclidean_axis_angle_fixed)
{
  EXPECT_NEAR((to_euclidean_element(Axis{}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Radians{}, g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Radians{}, g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Degrees{}, g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Degrees{}, g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis>{}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Axis>{}, g(3., 2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Axis>{}, g(3., 2.), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Dimensions<2>, Dimensions<3>>{}, g(3., 2., 5., 6., 7.), 4, 0)), 7., 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Degrees>{}, g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Degrees>{}, g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Radians, angle::PositiveRadians>{}, g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Radians, angle::PositiveRadians>{}, g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Radians, angle::Radians>{}, g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Radians, angle::Radians>{}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<angle::Radians, Dimensions<2>, angle::Radians>{}, g(pi / 3, 3., 4., pi / 6), 5, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, angle::Radians>{}, g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, angle::Radians>{}, g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, angle::Radians>{}, g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_axis_angle_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Axis{}}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Radians{}}, g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Radians{}}, g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Degrees{}}, g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Degrees{}}, g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis>{}}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Axis>{}}, g(3., 2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Axis>{}}, g(3., 2.), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Axis{}, DynamicTypedIndex {Axis{}}}, g(3., 2.), 1, 0)), 2., 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Degrees>{}}, g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Degrees>{}}, g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians, angle::PositiveRadians>{}}, g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians, angle::PositiveRadians>{}}, g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians, angle::Radians>{}}, g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians, angle::Radians>{}}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {DynamicTypedIndex {angle::Radians{}}, angle::Radians{}}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Radians{}, DynamicTypedIndex {angle::Radians{}}}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Radians{}, Dimensions<2>{}, DynamicTypedIndex {angle::Radians{}}}, g(pi / 3, 3., 4., pi / 6), 5, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Axis{}, angle::Radians{}}, g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g<float>(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Axis, angle::Radians>{}}, g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Axis, angle::Radians>{}}, g<float>(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Axis, angle::Radians>{}}, g<long double>(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_distance_inclination_fixed)
{
  EXPECT_NEAR((to_euclidean_element(Distance{}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(Distance{}, g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Radians{}, g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Radians{}, g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Degrees{}, g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(angle::Degrees{}, g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Distance>{}, g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Distance, Distance>{}, g(3., -2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Distance, Distance>{}, g(3., -2.), 1, 0)), -2., 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<inclination::Degrees>{}, g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<inclination::Degrees>{}, g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Distance, inclination::Radians>{}, g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Distance, inclination::Radians>{}, g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Distance, inclination::Radians>{}, g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_distance_inclination_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Distance{}}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Distance{}}, g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Radians{}}, g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Radians{}}, g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Degrees{}}, g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {angle::Degrees{}}, g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance>{}}, g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, Distance>{}}, g(3., -2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, Distance>{}}, g(3., -2.), 1, 0)), -2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Distance{}, DynamicTypedIndex{Distance{}}}, g(3., -2.), 1, 0)), -2., 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Degrees>{}}, g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Degrees>{}}, g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, inclination::Radians>{}}, g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, inclination::Radians>{}}, g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, inclination::Radians>{}}, g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, inclination::Radians>{}}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {DynamicTypedIndex{inclination::Radians{}}, inclination::Radians{}}, g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Distance{}, DynamicTypedIndex{inclination::Radians{}, Distance{}}}, g(3., pi / 3, 4), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Distance{}, DynamicTypedIndex{inclination::Radians{}, Distance{}}}, g(3., pi / 3, 4), 3, 0)), 4, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g<double>(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Distance, inclination::Radians>{}}, g<double>(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Distance, inclination::Radians>{}}, g<float>(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Distance, inclination::Radians>{}}, g<long double>(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_polar_fixed)
{
  EXPECT_NEAR((to_euclidean_element(Polar<Distance, angle::Radians>{}, g(3., pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<Distance, angle::Radians>{}, g(3., pi/3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<Distance, angle::Radians>{}, g(3., pi/3), 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<angle::Radians, Distance>{}, g(pi/3, 3.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<angle::Radians, Distance>{}, g(pi/3, 3.), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<angle::Radians, Distance>{}, g(pi/3, 3.), 2, 0)), 3., 1e-6);

  EXPECT_NEAR((to_euclidean_element(Polar<Distance, angle::Degrees>{}, g(3., 60.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<Distance, angle::Degrees>{}, g(3., 60.), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Polar<Distance, angle::Degrees>{}, g(3., 60.), 2, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 2, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
}


TEST(index_descriptors, toEuclidean_polar_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<Distance, angle::Radians>{}}, g(3., pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<Distance, angle::Radians>{}}, g(3., pi/3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<Distance, angle::Radians>{}}, g(3., pi/3), 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<angle::Radians, Distance>{}}, g(pi/3, 3.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<angle::Radians, Distance>{}}, g(pi/3, 3.), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<angle::Radians, Distance>{}}, g(pi/3, 3.), 2, 0)), 3., 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<Distance, angle::Degrees>{}}, g(3., 60.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<Distance, angle::Degrees>{}}, g(3., 60.), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Polar<Distance, angle::Degrees>{}}, g(3., 60.), 2, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g(1.1, 3., 60.), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g(1.1, 3., 60.), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g(1.1, 3., 60.), 2, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Axis{}, DynamicTypedIndex{Polar<Distance, angle::Degrees>{}}}, g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {DynamicTypedIndex{Axis{}}, Polar<Distance, angle::Degrees>{}}, g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g<long double>(1.1, 3., 60.), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g<double>(1.1, 3., 60.), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g<float>(1.1, 3., 60.), 2, 0)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}, g<long double>(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
}


TEST(index_descriptors, toEuclidean_spherical_fixed)
{
  EXPECT_NEAR((to_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 3, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 2, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 3, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 4, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
}


TEST(index_descriptors, toEuclidean_spherical_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., pi/6, pi/3), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., pi/6, pi/3), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., pi/6, pi/3), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., pi/6, pi/3), 3, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., pi/6, pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., pi/6, pi/3), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., pi/6, pi/3), 2, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., pi/6, pi/3), 3, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., pi/6, pi/3), 4, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., pi/6, pi/3, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., pi/6, pi/3, 3.), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., pi/6, pi/3, 3.), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., pi/6, pi/3, 3.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {DynamicTypedIndex{Spherical<Distance, angle::Radians, inclination::Radians>{}}, Axis{}}, g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}, DynamicTypedIndex{Axis{}}}, g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);

  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<float>(2., pi/6, pi/3, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<double>(2., pi/6, pi/3, 3.), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<float>(2., pi/6, pi/3, 3.), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<long double>(2., pi/6, pi/3, 3.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<long double>(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
}


TEST(index_descriptors, fromEuclidean_axis_angle_fixed)
{
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis>{}, g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis>{}, g(-3.), 0, 0)), -3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<angle::Radians>{}, g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<angle::Degrees>{}, g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, angle::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, angle::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<angle::Radians, Axis>{}, g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<angle::Radians, Axis>{}, g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_axis_angle_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis>{}}, g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis>{}}, g(-3.), 0, 0)), -3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians>{}}, g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Degrees>{}}, g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians, Axis>{}}, g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<angle::Radians, Axis>{}}, g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g<float>(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Axis, angle::Radians>{}}, g<double>(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Axis, angle::Radians>{}}, g<float>(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Axis, angle::Radians>{}}, g<long double>(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_distance_inclination_fixed)
{
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Distance>{}, g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Distance>{}, g(-3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<inclination::Radians>{}, g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<inclination::Degrees>{}, g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Distance, inclination::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Distance, inclination::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<inclination::Radians, Distance>{}, g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<inclination::Radians, Distance>{}, g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_distance_inclination_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Distance>{}}, g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Distance>{}}, g(-3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians>{}}, g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Degrees>{}}, g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, Distance>{}}, g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, Distance>{}}, g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<inclination::Radians, Distance>{}}, g<double>(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<double> {TypedIndex<inclination::Radians, Distance>{}}, g<double>(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<float> {TypedIndex<inclination::Radians, Distance>{}}, g<float>(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<inclination::Radians, Distance>{}}, g<long double>(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_polar_fixed)
{
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Polar<Distance, angle::Radians>>{}, g(3., 0.5, std::sqrt(3) / 2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Polar<Distance, angle::Radians>>{}, g(3., 0.5, std::sqrt(3) / 2), 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(0.5, std::sqrt(3)/2, 3, 1.1), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(0.5, std::sqrt(3)/2, 3, 1.1), 2, 0)), 1.1, 1e-6);
}


TEST(index_descriptors, fromEuclidean_polar_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::Radians>>{}}, g(3., 0.5, std::sqrt(3) / 2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::Radians>>{}}, g(3., 0.5, std::sqrt(3) / 2), 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(0.5, std::sqrt(3)/2, 3, 1.1), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(0.5, std::sqrt(3)/2, 3, 1.1), 2, 0)), 1.1, 1e-6);

  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g<long double>(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g<double>(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g<float>(0.5, std::sqrt(3)/2, 3, 1.1), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g<long double>(0.5, std::sqrt(3)/2, 3, 1.1), 2, 0)), 1.1, 1e-6);
}


TEST(index_descriptors, fromEuclidean_spherical_fixed)
{
  EXPECT_NEAR((from_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3, 0)), pi/3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
}


TEST(index_descriptors, fromEuclidean_spherical_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3, 0)), pi/3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {DynamicTypedIndex{Spherical<Distance, angle::Radians, inclination::Radians>{}}, Axis{}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}, DynamicTypedIndex{Axis{}}}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);

  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<float>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<double> {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<double>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<float> {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<float>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2, 0)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(DynamicTypedIndex<long double> {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}, g<long double>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
}


TEST(index_descriptors, wrap_get_element_axis_angle_fixed)
{
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis>{}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis>{}, g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians>{}, g(1.2*pi), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::PositiveRadians>{}, g(-0.2*pi), 0, 0)), 1.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Degrees>{}, g(200.), 0, 0)), -160., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::PositiveDegrees>{}, g(-20.), 0, 0)), 340., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Circle>{}, g(-0.2), 0, 0)), 0.8, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, angle::Radians>{}, g(3, 1.2*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, angle::Radians>{}, g(3, 1.2*pi), 1, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians, Axis>{}, g(1.2*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians, Axis>{}, g(1.2*pi, 3), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 1, 0)), 5, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 2, 0)), -0.7*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 3, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_axis_angle_dynamic)
{
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis>{}}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis>{}}, g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians>{}}, g(1.2*pi), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::PositiveRadians>{}}, g(-0.2*pi), 0, 0)), 1.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Degrees>{}}, g(200.), 0, 0)), -160., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::PositiveDegrees>{}}, g(-20.), 0, 0)), 340., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Circle>{}}, g(-0.2), 0, 0)), 0.8, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3, 1.2*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, g(3, 1.2*pi), 1, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, Axis>{}}, g(1.2*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, Axis>{}}, g(1.2*pi, 3), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g(1.2*pi, 5, 1.3*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g(1.2*pi, 5, 1.3*pi, 3), 1, 0)), 5, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g(1.2*pi, 5, 1.3*pi, 3), 2, 0)), -0.7*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g(1.2*pi, 5, 1.3*pi, 3), 3, 0)), 3, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g<double>(1.2*pi, 5, 1.3*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<double> {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g<double>(1.2*pi, 5, 1.3*pi, 3), 1, 0)), 5, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<float> {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g<float>(1.2*pi, 5, 1.3*pi, 3), 2, 0)), -0.7*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<long double> {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}, g<long double>(1.2*pi, 5, 1.3*pi, 3), 3, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_distance_inclination_fixed)
{
  EXPECT_NEAR((wrap_get_element(TypedIndex<Distance>{}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Distance>{}, g(-3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<inclination::Radians>{}, g(0.7*pi), 0, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<inclination::Radians>{}, g(1.2*pi), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<inclination::Degrees>{}, g(110.), 0, 0)), 70., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<inclination::Degrees>{}, g(200.), 0, 0)), -20., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Distance, inclination::Radians>{}, g(3, 0.7*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Distance, inclination::Radians>{}, g(3, 0.7*pi), 1, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<inclination::Radians, Distance>{}, g(1.2*pi, 3), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<inclination::Radians, Distance>{}, g(1.2*pi, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_distance_inclination_dynamic)
{
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Distance>{}}, g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Distance>{}}, g(-3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<inclination::Radians>{}}, g(0.7*pi), 0, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<inclination::Radians>{}}, g(1.2*pi), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<inclination::Degrees>{}}, g(110.), 0, 0)), 70., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<inclination::Degrees>{}}, g(200.), 0, 0)), -20., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3, 0.7*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g(3, 0.7*pi), 1, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<inclination::Radians, Distance>{}}, g(1.2*pi, 3), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<inclination::Radians, Distance>{}}, g(1.2*pi, 3), 1, 0)), 3, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Distance, inclination::Radians>{}}, g<long double>(3, 0.7*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<double> {TypedIndex<Distance, inclination::Radians>{}}, g<double>(3, 0.7*pi), 1, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<float> {TypedIndex<inclination::Radians, Distance>{}}, g<float>(1.2*pi, 3), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<long double> {TypedIndex<inclination::Radians, Distance>{}}, g<long double>(1.2*pi, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_polar_fixed)
{
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(3., -2., 1.1*pi), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(3., -2., 1.1*pi), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(3., 2., 1.1*pi), 2, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<Distance, angle::Radians>>{}, g(3., -2., 1.1*pi), 2, 0)), 0.1*pi, 1e-6);

  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(190., 2., 4.), 0, 0)), -170, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(190., -2., 4.), 0, 0)), 10, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(190., -2., 4.), 1, 0)), 2, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}, g(190., -2., -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(-2., -0.1*pi, 4.), 0, 0)), 2, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(2., -0.1*pi, 4.), 1, 0)), 1.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(-2., -0.1*pi, 4.), 1, 0)), 0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(-2., -0.1*pi, -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., 2.), 0, 0)), 5, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., 2.), 1, 0)), 350, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., -2.), 1, 0)), 170, 1e-6);
  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., -2.), 2, 0)), 2, 1e-6);
}


TEST(index_descriptors, wrap_get_element_polar_dynamic)
{
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(3., -2., 1.1*pi), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(3., -2., 1.1*pi), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(3., 2., 1.1*pi), 2, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}, g(3., -2., 1.1*pi), 2, 0)), 0.1*pi, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(190., 2., 4.), 0, 0)), -170, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(190., -2., 4.), 0, 0)), 10, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(190., -2., 4.), 1, 0)), 2, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}, g(190., -2., -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}, g(-2., -0.1*pi, 4.), 0, 0)), 2, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}, g(2., -0.1*pi, 4.), 1, 0)), 1.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}, g(-2., -0.1*pi, 4.), 1, 0)), 0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}, g(-2., -0.1*pi, -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g(5., -10., 2.), 0, 0)), 5, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g(5., -10., 2.), 1, 0)), 350, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g(5., -10., -2.), 1, 0)), 170, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g(5., -10., -2.), 2, 0)), 2, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g<float>(5., -10., 2.), 0, 0)), 5, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<double> {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g<double>(5., -10., 2.), 1, 0)), 350, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<float> {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g<float>(5., -10., -2.), 1, 0)), 170, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<long double> {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}, g<long double>(5., -10., -2.), 2, 0)), 2, 1e-6);
}


TEST(index_descriptors, wrap_get_element_spherical_fixed)
{
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 1.1*pi, 0.6*pi), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 0.5*pi, 0.6*pi), 1, 0)), -0.5*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 0.5*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 0.5*pi, 0.6*pi), 1, 0)), 0.5*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 0.5*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 1.1*pi, 0.6*pi), 1, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 1.1*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 1.1*pi, 0.6*pi), 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 1.1*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);

  EXPECT_NEAR((wrap_get_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
}


TEST(index_descriptors, wrap_get_element_spherical_dynamic)
{
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., 1.1*pi, 0.6*pi), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., 0.5*pi, 0.6*pi), 1, 0)), -0.5*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., 0.5*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(-2., 0.5*pi, 0.6*pi), 1, 0)), 0.5*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(-2., 0.5*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., 1.1*pi, 0.6*pi), 1, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(2., 1.1*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(-2., 1.1*pi, 0.6*pi), 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(-2., 1.1*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {DynamicTypedIndex{Axis{}}, Spherical<Distance, angle::Radians, inclination::Radians>{}}, g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Axis{}, DynamicTypedIndex{Spherical<Distance, angle::Radians, inclination::Radians>{}}}, g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);

  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g<float>(2., 1.1*pi, 0.6*pi), 1, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<double> {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g<double>(2., 1.1*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<float> {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g<float>(-2., 1.1*pi, 0.6*pi), 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((wrap_get_element(DynamicTypedIndex<long double> {Spherical<Distance, angle::Radians, inclination::Radians>{}}, g<long double>(-2., 1.1*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);
}


TEST(index_descriptors, wrap_set_element_axis_angle)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) { return a1[i]; }};

  wrap_set_element(TypedIndex<Axis, Axis>{}, s, g, 2.1, 1, 0);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  wrap_set_element(TypedIndex<Axis, angle::Radians>{}, s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  wrap_set_element(TypedIndex<angle::PositiveRadians, Axis>{}, s, g, -0.2*pi, 0, 0);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  wrap_set_element(TypedIndex<angle::Degrees, Axis>{}, s, g, 200., 0, 0);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  wrap_set_element(TypedIndex<Axis, angle::PositiveDegrees>{}, s, g, -20., 1, 0);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  wrap_set_element(TypedIndex<Axis, angle::Circle>{}, s, g, -0.2, 1, 0);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);

  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Axis>{}}, s, g, 2.1, 1, 0);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<angle::PositiveRadians, Axis>{}}, s, g, -0.2*pi, 0, 0);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<angle::Degrees, Axis>{}}, s, g, 200., 0, 0);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, angle::PositiveDegrees>{}}, s, g, -20., 1, 0);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, angle::Circle>{}}, s, g, -0.2, 1, 0);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);

  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Axis>{}}, s, g, 2.1, 1, 0);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, angle::Radians>{}}, s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<angle::PositiveRadians, Axis>{}}, s, g, -0.2*pi, 0, 0);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<angle::Degrees, Axis>{}}, s, g, 200., 0, 0);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, angle::PositiveDegrees>{}}, s, g, -20., 1, 0);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, angle::Circle>{}}, s, g, -0.2, 1, 0);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);
}


TEST(index_descriptors, wrap_set_element_distance_inclination)
{
  std::array<float, 2> a1 = {0, 0};
  auto s = std::function {[&](const float& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  wrap_set_element(TypedIndex<Distance, Distance>{}, s, g, 2.1, 0, 0);
  wrap_set_element(TypedIndex<Distance, Distance>{}, s, g, -2.1, 1, 0);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  wrap_set_element(TypedIndex<inclination::Radians, inclination::Degrees>{}, s, g, 0.7*pi, 0, 0);
  wrap_set_element(TypedIndex<inclination::Radians, inclination::Degrees>{}, s, g, 110., 1, 0);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  wrap_set_element(TypedIndex<inclination::Degrees, inclination::Radians>{}, s, g, 200., 0, 0);
  wrap_set_element(TypedIndex<inclination::Degrees, inclination::Radians>{}, s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);

  wrap_set_element(DynamicTypedIndex {TypedIndex<Distance, Distance>{}}, s, g, 2.1, 0, 0);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Distance, Distance>{}}, s, g, -2.1, 1, 0);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<inclination::Radians, inclination::Degrees>{}}, s, g, 0.7*pi, 0, 0);
  wrap_set_element(DynamicTypedIndex {TypedIndex<inclination::Radians, inclination::Degrees>{}}, s, g, 110., 1, 0);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<inclination::Degrees, inclination::Radians>{}}, s, g, 200., 0, 0);
  wrap_set_element(DynamicTypedIndex {TypedIndex<inclination::Degrees, inclination::Radians>{}}, s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);

  wrap_set_element(DynamicTypedIndex<float> {TypedIndex<Distance, Distance>{}}, s, g, 2.1, 0, 0);
  wrap_set_element(DynamicTypedIndex<float> {TypedIndex<Distance, Distance>{}}, s, g, -2.1, 1, 0);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  wrap_set_element(DynamicTypedIndex<float> {TypedIndex<inclination::Radians, inclination::Degrees>{}}, s, g, 0.7*pi, 0, 0);
  wrap_set_element(DynamicTypedIndex<float> {TypedIndex<inclination::Radians, inclination::Degrees>{}}, s, g, 110., 1, 0);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  wrap_set_element(DynamicTypedIndex<float> {TypedIndex<inclination::Degrees, inclination::Radians>{}}, s, g, 200., 0, 0);
  wrap_set_element(DynamicTypedIndex<float> {TypedIndex<inclination::Degrees, inclination::Radians>{}}, s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
}


TEST(index_descriptors, wrap_set_element_polar_fixed)
{
  std::array<double, 4> a1 = {1, 0.8*pi, 0, 1};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  wrap_set_element(TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
  wrap_set_element(TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, 1.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.9*pi, 1e-6);
  wrap_set_element(TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, 190., 2, 0);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  wrap_set_element(TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  wrap_set_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -0.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], 1.9*pi, 1e-6);
  wrap_set_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -10., 2, 0);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  wrap_set_element(TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(index_descriptors, wrap_set_element_polar_dynamic)
{
  std::array<long double, 4> a1 = {1, -0.2*numbers::pi_v<long double>, 0, 1};
  auto s = std::function {[&](const long double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}, s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  EXPECT_NEAR(a1[1], 0.8*numbers::pi_v<long double>, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}, s, g, 1.1*numbers::pi_v<long double>, 1, 0);
  EXPECT_NEAR(a1[1], -0.9*numbers::pi_v<long double>, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}, s, g, 190., 2, 0);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}, s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -0.1*numbers::pi_v<long double>, 1, 0);
  EXPECT_NEAR(a1[1], 1.9*numbers::pi_v<long double>, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -10., 2, 0);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(DynamicTypedIndex<long double> {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  wrap_set_element(DynamicTypedIndex<long double> {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -0.1*numbers::pi_v<long double>, 1, 0);
  EXPECT_NEAR(a1[1], 1.9*numbers::pi_v<long double>, 1e-6);
  wrap_set_element(DynamicTypedIndex<long double> {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -10., 2, 0);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  wrap_set_element(DynamicTypedIndex<long double> {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}, s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(index_descriptors, wrap_set_element_spherical_fixed)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, 1.1*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, 0.6*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -0.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -2.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, 0.6*pi, 2, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, 1.1*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, 3.4*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -2., 2, 0);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -170., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, 100., 3, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, 550., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  wrap_set_element(TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -3., 2, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}


TEST(index_descriptors, wrap_set_element_spherical_dynamic)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, 1.1*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, 0.6*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, -0.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}, s, g, -2.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}, s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}, s, g, 0.6*pi, 2, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}, s, g, 1.1*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}, s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}, s, g, 3.4*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -2., 2, 0);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -170., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, 100., 3, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, 550., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  wrap_set_element(DynamicTypedIndex {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -3., 2, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);

  a1 = {0, 0, 0, 0};

  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -2., 2, 0);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -170., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, 100., 3, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, 550., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  wrap_set_element(DynamicTypedIndex<double> {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}, s, g, -3., 2, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}

