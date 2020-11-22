/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "coefficients.hpp"


using std::numbers::pi;

template<typename...Ss>
inline auto g(const Ss...ss)
{
  using Scalar = std::common_type_t<Ss...>;
  using GetCoeff = std::function<Scalar(const std::size_t)>;
  auto f = [=](const std::size_t i)
  {
    auto arr = std::array<Scalar, sizeof...(Ss)> {static_cast<Scalar>(ss)...};
    return arr[i];
  };
  return GetCoeff {f};
}


TEST_F(coefficients, toEuclidean_axis_angle)
{
  EXPECT_NEAR((Axis::to_Euclidean_array<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((angle::Radians::to_Euclidean_array<double, 0>[0](g(pi/3))), 0.5, 1e-6);
  EXPECT_NEAR((angle::Radians::to_Euclidean_array<double, 0>[1](g(pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Axis, double>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<angle::Degrees, double>(0, g(60))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<angle::Degrees, double>(1, g(30))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis>, double>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Axis>, double>(0, g(3., 2.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Axis>, double>(1, g(3., 2.))), 2., 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<angle::Degrees>, double>(0, g(-60))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<angle::Degrees>, double>(1, g(-30))), -0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<angle::Radians, angle::PositiveRadians>, double>(0, g(pi / 3, pi / 6))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<angle::Radians, angle::PositiveRadians>, double>(1, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<angle::Radians, angle::Radians>, double>(2, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<angle::Radians, angle::Radians>, double>(3, g(pi / 3, pi / 6))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, angle::Radians>, double>(0, g(3., pi / 3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, angle::Radians>, double>(1, g(3., pi / 3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, angle::Radians>, double>(2, g(3., pi / 3))), std::sqrt(3) / 2, 1e-6);
}


TEST_F(coefficients, toEuclidean_distance_inclination)
{
  EXPECT_NEAR((Distance::to_Euclidean_array<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Distance::to_Euclidean_array<double, 0>[0](g(-3.))), -3., 1e-6);
  EXPECT_NEAR((inclination::Radians::to_Euclidean_array<double, 0>[0](g(pi/3))), 0.5, 1e-6);
  EXPECT_NEAR((inclination::Radians::to_Euclidean_array<double, 0>[1](g(pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Distance, double>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<angle::Degrees, double>(0, g(60))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<angle::Degrees, double>(1, g(30))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Distance>, double>(0, g(-3.))), -3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Distance, Distance>, double>(0, g(3., -2.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Distance, Distance>, double>(1, g(3., -2.))), -2., 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<inclination::Degrees>, double>(0, g(-60))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<inclination::Degrees>, double>(1, g(-30))), -0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<inclination::Radians, inclination::Radians>, double>(0, g(pi / 3, pi / 6))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<inclination::Radians, inclination::Radians>, double>(1, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<inclination::Radians, inclination::Radians>, double>(2, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<inclination::Radians, inclination::Radians>, double>(3, g(pi / 3, pi / 6))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Distance, inclination::Radians>, double>(0, g(3., pi / 3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Distance, inclination::Radians>, double>(1, g(3., pi / 3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Distance, inclination::Radians>, double>(2, g(3., pi / 3))), std::sqrt(3) / 2, 1e-6);
}


TEST_F(coefficients, toEuclidean_polar)
{
  EXPECT_NEAR((Polar<Distance, angle::Radians>::to_Euclidean_array<double, 0>[0](g(3., pi/3))), 3., 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::to_Euclidean_array<double, 0>[1](g(3., pi/3))), 0.5, 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::to_Euclidean_array<double, 0>[2](g(3., pi/3))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((Polar<angle::Radians, Distance>::to_Euclidean_array<double, 0>[0](g(pi/3, 3.))), 0.5, 1e-6);
  EXPECT_NEAR((Polar<angle::Radians, Distance>::to_Euclidean_array<double, 0>[1](g(pi/3, 3.))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((Polar<angle::Radians, Distance>::to_Euclidean_array<double, 0>[2](g(pi/3, 3.))), 3., 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Polar<Distance, angle::Degrees>, double>(0, g(3., 60.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Polar<Distance, angle::Degrees>, double>(1, g(3., 60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Polar<Distance, angle::Degrees>, double>(2, g(3., 60.))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Polar<Distance, angle::Degrees>>, double>(0, g(1.1, 3., 60.))), 1.1, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Polar<Distance, angle::Degrees>>, double>(1, g(1.1, 3., 60.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Polar<Distance, angle::Degrees>>, double>(2, g(1.1, 3., 60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Polar<Distance, angle::Degrees>>, double>(3, g(1.1, 3., 60.))), std::sqrt(3)/2, 1e-6);
}


TEST_F(coefficients, toEuclidean_spherical)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_Euclidean_array<double, 0>[0](g(2., pi/6, pi/3))), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_Euclidean_array<double, 0>[1](g(2., pi/6, pi/3))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_Euclidean_array<double, 0>[2](g(2., pi/6, pi/3))), 0.25, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_Euclidean_array<double, 0>[3](g(2., pi/6, pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((Spherical<angle::Degrees, Distance, inclination::Degrees>::to_Euclidean_array<double, 0>[0](g(30, 2., 60))), 2., 1e-6);
  EXPECT_NEAR((Spherical<angle::Degrees, Distance, inclination::Degrees>::to_Euclidean_array<double, 0>[1](g(30, 2., 60))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((Spherical<angle::Degrees, Distance, inclination::Degrees>::to_Euclidean_array<double, 0>[2](g(30, 2., 60))), 0.25, 1e-6);
  EXPECT_NEAR((Spherical<angle::Degrees, Distance, inclination::Degrees>::to_Euclidean_array<double, 0>[3](g(30, 2., 60))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(0, g(2., pi/6, pi/3))), 2., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(1, g(2., pi/6, pi/3))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(2, g(2., pi/6, pi/3))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(3, g(2., pi/6, pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(0, g(3., 2., pi/6, pi/3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(1, g(3., 2., pi/6, pi/3))), 2., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(2, g(3., 2., pi/6, pi/3))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(3, g(3., 2., pi/6, pi/3))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(4, g(3., 2., pi/6, pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(0, g(2., pi/6, pi/3, 3.))), 2., 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(1, g(2., pi/6, pi/3, 3.))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(2, g(2., pi/6, pi/3, 3.))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(3, g(2., pi/6, pi/3, 3.))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(4, g(2., pi/6, pi/3, 3.))), 3., 1e-6);
}


TEST_F(coefficients, fromEuclidean_axis_angle)
{
  EXPECT_NEAR((Axis::from_Euclidean_array<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Axis::from_Euclidean_array<double, 0>[0](g(-3.))), -3., 1e-6);
  EXPECT_NEAR((angle::Radians::from_Euclidean_array<double, 0>[0](g(0.5, std::sqrt(3) / 2))), pi / 3, 1e-6);
  EXPECT_NEAR((angle::Degrees::from_Euclidean_array<double, 0>[0](g(0.5, std::sqrt(3) / 2))), 60, 1e-6);

  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis>, double>(0, g(3.))), 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<angle::Radians>, double>(0, g(0.5, std::sqrt(3) / 2))), pi / 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, angle::Radians>, double>(0, g(3, 0.5, -std::sqrt(3) / 2))), 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, angle::Radians>, double>(1, g(3, 0.5, -std::sqrt(3) / 2))), -pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<angle::Radians, Axis>, double>(0, g(0.5, std::sqrt(3) / 2, 3))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<angle::Radians, Axis>, double>(1, g(0.5, std::sqrt(3) / 2, 3))), 3, 1e-6);
}


TEST_F(coefficients, fromEuclidean_distance_inclination)
{
  EXPECT_NEAR((Distance::from_Euclidean_array<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Distance::from_Euclidean_array<double, 0>[0](g(-3.))), 3., 1e-6);
  EXPECT_NEAR((inclination::Radians::from_Euclidean_array<double, 0>[0](g(0.5, std::sqrt(3) / 2))), pi / 3, 1e-6);
  EXPECT_NEAR((inclination::Degrees::from_Euclidean_array<double, 0>[0](g(0.5, std::sqrt(3) / 2))), 60, 1e-6);

  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Distance>, double>(0, g(3.))), 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<inclination::Radians>, double>(0, g(0.5, std::sqrt(3) / 2))), pi / 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Distance, inclination::Radians>, double>(0, g(3, 0.5, -std::sqrt(3) / 2))), 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Distance, inclination::Radians>, double>(1, g(3, 0.5, -std::sqrt(3) / 2))), -pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<inclination::Radians, Distance>, double>(0, g(0.5, std::sqrt(3) / 2, 3))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<inclination::Radians, Distance>, double>(1, g(0.5, std::sqrt(3) / 2, 3))), 3, 1e-6);
}


TEST_F(coefficients, fromEuclidean_polar)
{
  EXPECT_NEAR((Polar<Distance, angle::Radians>::from_Euclidean_array<double, 0>[0](g(3., 0.5, std::sqrt(3) / 2))), 3., 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::from_Euclidean_array<double, 0>[1](g(3., 0.5, std::sqrt(3) / 2))), pi/3, 1e-6);
  EXPECT_NEAR((Polar<angle::Degrees, Distance>::from_Euclidean_array<double, 0>[0](g(0.5, std::sqrt(3)/2, 3.))), 60., 1e-6);
  EXPECT_NEAR((Polar<angle::Degrees, Distance>::from_Euclidean_array<double, 0>[1](g(0.5, std::sqrt(3)/2, 3.))), 3., 1e-6);

  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Polar<Distance, angle::Radians>>, double>(0, g(3., 0.5, std::sqrt(3) / 2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Polar<Distance, angle::Radians>>, double>(1, g(3., 0.5, std::sqrt(3) / 2))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Polar<Distance, angle::Radians>>, double>(0, g(1.1, 3, 0.5, -std::sqrt(3) / 2))), 1.1, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Polar<Distance, angle::Radians>>, double>(1, g(1.1, 3, 0.5, -std::sqrt(3) / 2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Polar<Distance, angle::Radians>>, double>(2, g(1.1, 3, 0.5, -std::sqrt(3) / 2))), -pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Polar<angle::Degrees, Distance>, Axis>, double>(0, g(0.5, std::sqrt(3)/2, 3, 1.1))), 60, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Polar<angle::Degrees, Distance>, Axis>, double>(1, g(0.5, std::sqrt(3)/2, 3, 1.1))), 3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Polar<angle::Degrees, Distance>, Axis>, double>(2, g(0.5, std::sqrt(3)/2, 3, 1.1))), 1.1, 1e-6);
}


TEST_F(coefficients, fromEuclidean_spherical)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_Euclidean_array<double, 0>[0](
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_Euclidean_array<double, 0>[1](
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/6, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_Euclidean_array<double, 0>[2](
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/3, 1e-6);

  EXPECT_NEAR((Spherical<Distance, angle::Degrees, inclination::Degrees>::from_Euclidean_array<double, 0>[0](
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Degrees, inclination::Degrees>::from_Euclidean_array<double, 0>[1](
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 30., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Degrees, inclination::Degrees>::from_Euclidean_array<double, 0>[2](
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 60., 1e-6);

  EXPECT_NEAR((internal::from_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(0,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(1,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Spherical<Distance, angle::Radians, inclination::Radians>, double>(2,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/3, 1e-6);

  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(0,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(1,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(2,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>, double>(3,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/3, 1e-6);

  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(0,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), 2., 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(1,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(2,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_Euclidean<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>, double>(3,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), 3., 1e-6);
}


TEST_F(coefficients, wrap_get_axis_angle)
{
  EXPECT_NEAR((Axis::wrap_array_get<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Axis::wrap_array_get<double, 0>[0](g(-3.))), -3., 1e-6);
  EXPECT_NEAR((angle::Radians::wrap_array_get<double, 0>[0](g(1.1*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((angle::PositiveRadians::wrap_array_get<double, 0>[0](g(-0.1*pi))), 1.9*pi, 1e-6);
  EXPECT_NEAR((angle::Degrees::wrap_array_get<double, 0>[0](g(190.))), -170., 1e-6);
  EXPECT_NEAR((angle::PositiveDegrees::wrap_array_get<double, 0>[0](g(-10.))), 350., 1e-6);
  EXPECT_NEAR((angle::Circle::wrap_array_get<double, 0>[0](g(-0.1))), 0.9, 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis>>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis>>(0, g(-3.))), -3., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::Radians>>(0, g(1.2*pi))), -0.8*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::PositiveRadians>>(0, g(-0.2*pi))), 1.8*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::Degrees>>(0, g(200.))), -160., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::PositiveDegrees>>(0, g(-20.))), 340., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::Circle>>(0, g(-0.2))), 0.8, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, angle::Radians>>(0, g(3, 1.2*pi))), 3, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, angle::Radians>>(1, g(3, 1.2*pi))), -0.8*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::Radians, Axis>>(0, g(1.2*pi, 3))), -0.8*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<angle::Radians, Axis>>(1, g(1.2*pi, 3))), 3, 1e-6);
}


TEST_F(coefficients, wrap_get_distance_inclination)
{
  EXPECT_NEAR((Distance::wrap_array_get<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Distance::wrap_array_get<double, 0>[0](g(-3.))), 3., 1e-6);
  EXPECT_NEAR((inclination::Radians::wrap_array_get<double, 0>[0](g(0.6*pi))), 0.4*pi, 1e-6);
  EXPECT_NEAR((inclination::Radians::wrap_array_get<double, 0>[0](g(1.1*pi))), -0.1*pi, 1e-6);
  EXPECT_NEAR((inclination::Degrees::wrap_array_get<double, 0>[0](g(100.))), 80., 1e-6);
  EXPECT_NEAR((inclination::Degrees::wrap_array_get<double, 0>[0](g(190.))), -10., 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Distance>>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Distance>>(0, g(-3.))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<inclination::Radians>>(0, g(0.7*pi))), 0.3*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<inclination::Radians>>(0, g(1.2*pi))), -0.2*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<inclination::Degrees>>(0, g(110.))), 70., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<inclination::Degrees>>(0, g(200.))), -20., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Distance, inclination::Radians>>(0, g(3, 0.7*pi))), 3, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Distance, inclination::Radians>>(1, g(3, 0.7*pi))), 0.3*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<inclination::Radians, Distance>>(0, g(1.2*pi, 3))), -0.2*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<inclination::Radians, Distance>>(1, g(1.2*pi, 3))), 3, 1e-6);
}


TEST_F(coefficients, wrap_get_polar)
{
  EXPECT_NEAR((Polar<Distance, angle::Radians>::wrap_array_get<double, 0>[0](g(-2., 1.1*pi))), 2., 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::wrap_array_get<double, 0>[1](g(2., 1.1*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::Radians>::wrap_array_get<double, 0>[1](g(-2., 1.1*pi))), 0.1*pi, 1e-6);

  EXPECT_NEAR((Polar<angle::Degrees, Distance>::wrap_array_get<double, 0>[0](g(190, 2.))), -170, 1e-6);
  EXPECT_NEAR((Polar<angle::Degrees, Distance>::wrap_array_get<double, 0>[0](g(190, -2.))), 10, 1e-6);
  EXPECT_NEAR((Polar<angle::Degrees, Distance>::wrap_array_get<double, 0>[1](g(190, -2.))), 2., 1e-6);

  EXPECT_NEAR((Polar<Distance, angle::PositiveRadians>::wrap_array_get<double, 0>[0](g(-2., -0.1*pi))), 2., 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::PositiveRadians>::wrap_array_get<double, 0>[1](g(2., -0.1*pi))), 1.9*pi, 1e-6);
  EXPECT_NEAR((Polar<Distance, angle::PositiveRadians>::wrap_array_get<double, 0>[1](g(-2., -0.1*pi))), 0.9*pi, 1e-6);

  EXPECT_NEAR((Polar<angle::PositiveDegrees, Distance>::wrap_array_get<double, 0>[0](g(-10, 2.))), 350, 1e-6);
  EXPECT_NEAR((Polar<angle::PositiveDegrees, Distance>::wrap_array_get<double, 0>[0](g(-10, -2.))), 170, 1e-6);
  EXPECT_NEAR((Polar<angle::PositiveDegrees, Distance>::wrap_array_get<double, 0>[1](g(-10, -2.))), 2., 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(0, g(3., -2., 1.1*pi))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(1, g(3., -2., 1.1*pi))), 2., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(2, g(3., 2., 1.1*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(2, g(3., -2., 1.1*pi))), 0.1*pi, 1e-6);
}


TEST_F(coefficients, wrap_get_spherical)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_array_get<double, 0>[0](
    g(-2., 1.1*pi, 0.3*pi))), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_array_get<double, 0>[1](
    g(2., 1.1*pi, 0.3*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_array_get<double, 0>[1](
    g(2., 1.1*pi, -0.3*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_array_get<double, 0>[1](
    g(-2., 1.1*pi, 0.3*pi))), 0.1*pi, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::wrap_array_get<double, 0>[1](
    g(-2., 1.1*pi, -0.3*pi))), 0.1*pi, 1e-6);

  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(0,
    g(2., 1.1*pi, 0.6*pi))), 2., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(1,
    g(2., 0.5*pi, 0.6*pi))), -0.5*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(2,
    g(2., 0.5*pi, 0.6*pi))), 0.4*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(1,
    g(-2., 0.5*pi, 0.6*pi))), 0.5*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(2,
    g(-2., 0.5*pi, 0.6*pi))), -0.4*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(1,
    g(2., 1.1*pi, 0.6*pi))), 0.1*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(2,
    g(2., 1.1*pi, 0.6*pi))), 0.4*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(1,
    g(-2., 1.1*pi, 0.6*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Spherical<Distance, angle::Radians, inclination::Radians>>(2,
    g(-2., 1.1*pi, 0.6*pi))), -0.4*pi, 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(3,
    g(3., -2., 1.1*pi, 0.6*pi))), -0.4*pi, 1e-6);
}


