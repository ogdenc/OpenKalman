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

#include "interfaces/eigen3/tests/eigen3.gtest.hpp"

using namespace OpenKalman;

using std::numbers::pi;

template<typename...Ss>
inline auto g(const Ss...ss)
{
  using Scalar = std::common_type_t<Ss...>;
  return [=](std::size_t i) {
    auto arr = std::array<Scalar, sizeof...(Ss)> {static_cast<Scalar>(ss)...};
    return arr[i];
  };
}


TEST(index_descriptors, toEuclidean_axis_angle)
{
  static_assert(index_descriptor<Axis>);
  EXPECT_NEAR((Axis::template to_euclidean_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((angle::Radians::template to_euclidean_element<double>(g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((Coefficients<Axis>::to_euclidean_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Axis>::to_euclidean_element<double>(g(3., 2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Axis>::to_euclidean_element<double>(g(3., 2.), 1, 0)), 2., 1e-6);

  EXPECT_NEAR((Coefficients<angle::Degrees>::to_euclidean_element<double>(g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Degrees>::to_euclidean_element<double>(g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, angle::PositiveRadians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, angle::PositiveRadians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, angle::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, angle::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);

  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::to_euclidean_element<double>(g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::to_euclidean_element<double>(g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::to_euclidean_element<double>(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_distance_inclination)
{
  EXPECT_NEAR((Distance::to_euclidean_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Distance::to_euclidean_element<double>(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((Coefficients<Distance>::to_euclidean_element<double>(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((Coefficients<Distance, Distance>::to_euclidean_element<double>(g(3., -2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Distance, Distance>::to_euclidean_element<double>(g(3., -2.), 1, 0)), -2., 1e-6);

  EXPECT_NEAR((Coefficients<inclination::Degrees>::to_euclidean_element<double>(g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Degrees>::to_euclidean_element<double>(g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);

  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::to_euclidean_element<double>(g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::to_euclidean_element<double>(g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::to_euclidean_element<double>(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_polar)
{
  EXPECT_NEAR((Polar<Distance, angle::Radians>::to_euclidean_element<double>(g(3., pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::to_euclidean_element<double>(g(3., pi/3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::to_euclidean_element<double>(g(3., pi/3), 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((Polar<angle::Radians, Distance>::to_euclidean_element<double>(g(pi/3, 3.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Polar<angle::Radians, Distance>::to_euclidean_element<double>(g(pi/3, 3.), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((Polar<angle::Radians, Distance>::to_euclidean_element<double>(g(pi/3, 3.), 2, 0)), 3., 1e-6);

  EXPECT_NEAR((Polar<Distance, angle::Degrees>::to_euclidean_element<double>(g(3., 60.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Degrees>::to_euclidean_element<double>(g(3., 60.), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Degrees>::to_euclidean_element<double>(g(3., 60.), 2, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 2, 0)), 0.5, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
}


TEST(index_descriptors, toEuclidean_spherical)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 3, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 2, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 3, 0)), 0.25, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 4, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
}


TEST(index_descriptors, fromEuclidean_axis_angle)
{
  EXPECT_NEAR((Coefficients<Axis>::from_euclidean_element<double>(g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Axis>::from_euclidean_element<double>(g(-3.), 0, 0)), -3, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Degrees>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_distance_inclination)
{
  EXPECT_NEAR((Coefficients<Distance>::from_euclidean_element<double>(g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Distance>::from_euclidean_element<double>(g(-3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Degrees>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, Distance>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, Distance>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_polar)
{
  EXPECT_NEAR((Coefficients<Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(3., 0.5, std::sqrt(3) / 2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(3., 0.5, std::sqrt(3) / 2), 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3)/2, 3, 1.1), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3)/2, 3, 1.1), 2, 0)), 1.1, 1e-6);
}


TEST(index_descriptors, fromEuclidean_spherical)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/3, 1e-6);

  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/6, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3, 0)), pi/3, 1e-6);

  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2, 0)), pi/3, 1e-6);
  EXPECT_NEAR((Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
}


TEST(index_descriptors, wrap_get_element_axis_angle)
{
  EXPECT_NEAR((Coefficients<Axis>::wrap_get_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis>::wrap_get_element<double>(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians>::wrap_get_element<double>(g(1.2*pi), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((Coefficients<angle::PositiveRadians>::wrap_get_element<double>(g(-0.2*pi), 0, 0)), 1.8*pi, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Degrees>::wrap_get_element<double>(g(200.), 0, 0)), -160., 1e-6);
  EXPECT_NEAR((Coefficients<angle::PositiveDegrees>::wrap_get_element<double>(g(-20.), 0, 0)), 340., 1e-6);
  EXPECT_NEAR((Coefficients<angle::Circle>::wrap_get_element<double>(g(-0.2), 0, 0)), 0.8, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::wrap_get_element<double>(g(3, 1.2*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, angle::Radians>::wrap_get_element<double>(g(3, 1.2*pi), 1, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, Axis>::wrap_get_element<double>(g(1.2*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((Coefficients<angle::Radians, Axis>::wrap_get_element<double>(g(1.2*pi, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_distance_inclination)
{
  EXPECT_NEAR((Coefficients<Distance>::wrap_get_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Distance>::wrap_get_element<double>(g(-3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians>::wrap_get_element<double>(g(0.7*pi), 0, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians>::wrap_get_element<double>(g(1.2*pi), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Degrees>::wrap_get_element<double>(g(110.), 0, 0)), 70., 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Degrees>::wrap_get_element<double>(g(200.), 0, 0)), -20., 1e-6);
  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::wrap_get_element<double>(g(3, 0.7*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((Coefficients<Distance, inclination::Radians>::wrap_get_element<double>(g(3, 0.7*pi), 1, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, Distance>::wrap_get_element<double>(g(1.2*pi, 3), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((Coefficients<inclination::Radians, Distance>::wrap_get_element<double>(g(1.2*pi, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_polar)
{
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., 2., 1.1*pi), 2, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi), 2, 0)), 0.1*pi, 1e-6);

  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., 2., 4.), 0, 0)), -170, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., -2., 4.), 0, 0)), 10, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., -2., 4.), 1, 0)), 2, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., -2., -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(-2., -0.1*pi, 4.), 0, 0)), 2, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(2., -0.1*pi, 4.), 1, 0)), 1.9*pi, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(-2., -0.1*pi, 4.), 1, 0)), 0.9*pi, 1e-6);
  EXPECT_NEAR((Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(-2., -0.1*pi, -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., 2.), 0, 0)), 5, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., 2.), 1, 0)), 350, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., -2.), 1, 0)), 170, 1e-6);
  EXPECT_NEAR((Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., -2.), 2, 0)), 2, 1e-6);
}


TEST(index_descriptors, wrap_get_element_spherical)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(2., 1.1*pi, 0.6*pi), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(2., 0.5*pi, 0.6*pi), 1, 0)), -0.5*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(2., 0.5*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(-2., 0.5*pi, 0.6*pi), 1, 0)), 0.5*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(-2., 0.5*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(2., 1.1*pi, 0.6*pi), 1, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(2., 1.1*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(-2., 1.1*pi, 0.6*pi), 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_get_element<double>(g(-2., 1.1*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);

  EXPECT_NEAR((Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
}


TEST(index_descriptors, wrap_set_element_axis_angle)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = [&](const std::size_t i, double s) {a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  Coefficients<Axis, Axis>::wrap_set_element<double>(s, g, 2.1, 1, 0);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  Coefficients<Axis, angle::Radians>::wrap_set_element<double>(s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  Coefficients<angle::PositiveRadians, Axis>::wrap_set_element<double>(s, g, -0.2*pi, 0, 0);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  Coefficients<angle::Degrees, Axis>::wrap_set_element<double>(s, g, 200., 0, 0);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  Coefficients<Axis, angle::PositiveDegrees>::wrap_set_element<double>(s, g, -20., 1, 0);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  Coefficients<Axis, angle::Circle>::wrap_set_element<double>(s, g, -0.2, 1, 0);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);
}


TEST(index_descriptors, wrap_set_element_distance_inclination)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = [&](const std::size_t i, double s) {a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  Coefficients<Distance, Distance>::wrap_set_element<double>(s, g, 2.1, 0, 0);
  Coefficients<Distance, Distance>::wrap_set_element<double>(s, g, -2.1, 1, 0);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  Coefficients<inclination::Radians, inclination::Degrees>::wrap_set_element<double>(s, g, 0.7*pi, 0, 0);
  Coefficients<inclination::Radians, inclination::Degrees>::wrap_set_element<double>(s, g, 110., 1, 0);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  Coefficients<inclination::Degrees, inclination::Radians>::wrap_set_element<double>(s, g, 200., 0, 0);
  Coefficients<inclination::Degrees, inclination::Radians>::wrap_set_element<double>(s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
}


TEST(index_descriptors, wrap_set_element_polar)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = [&](const std::size_t i, double s) {a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, 1.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.9*pi, 1e-6);
  Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, 190., 2, 0);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -0.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], 1.9*pi, 1e-6);
  Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -10., 2, 0);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(index_descriptors, wrap_set_element_spherical)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = [&](const std::size_t i, double s) {a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, 1.1*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, 0.6*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -0.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -2.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, 0.6*pi, 2, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, 1.1*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, 3.4*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -2., 2, 0);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -170., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, 100., 3, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, 550., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -3., 2, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}
