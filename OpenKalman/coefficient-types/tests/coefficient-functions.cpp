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
#include "coefficient-types/coefficient-types.hpp"


using namespace OpenKalman;

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


TEST(coefficients, toEuclidean_axis_angle)
{
  EXPECT_NEAR((internal::to_euclidean_coeff<Axis>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Radians>(0, g(pi/3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Radians>(1, g(pi/3))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Degrees>(0, g(60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Degrees>(1, g(30.))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis>>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Axis>>(0, g(3., 2.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Axis>>(1, g(3., 2.))), 2., 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<angle::Degrees>>(0, g(-60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<angle::Degrees>>(1, g(-30.))), -0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<angle::Radians, angle::PositiveRadians>>(0, g(pi / 3, pi / 6))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<angle::Radians, angle::PositiveRadians>>(1, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<angle::Radians, angle::Radians>>(2, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<angle::Radians, angle::Radians>>(3, g(pi / 3, pi / 6))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, angle::Radians>>(0, g(3., pi / 3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, angle::Radians>>(1, g(3., pi / 3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, angle::Radians>>(2, g(3., pi / 3))), std::sqrt(3) / 2, 1e-6);
}


TEST(coefficients, toEuclidean_distance_inclination)
{
  EXPECT_NEAR((internal::to_euclidean_coeff<Distance>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Distance>(0, g(-3.))), -3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Radians>(0, g(pi/3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Radians>(1, g(pi/3))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Degrees>(0, g(60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<angle::Degrees>(1, g(30.))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Distance>>(0, g(-3.))), -3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Distance, Distance>>(0, g(3., -2.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Distance, Distance>>(1, g(3., -2.))), -2., 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<inclination::Degrees>>(0, g(-60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<inclination::Degrees>>(1, g(-30.))), -0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<inclination::Radians, inclination::Radians>>(0, g(pi / 3, pi / 6))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<inclination::Radians, inclination::Radians>>(1, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<inclination::Radians, inclination::Radians>>(2, g(pi / 3, pi / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<inclination::Radians, inclination::Radians>>(3, g(pi / 3, pi / 6))), 0.5, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Distance, inclination::Radians>>(0, g(3., pi / 3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Distance, inclination::Radians>>(1, g(3., pi / 3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Distance, inclination::Radians>>(2, g(3., pi / 3))), std::sqrt(3) / 2, 1e-6);
}


TEST(coefficients, toEuclidean_polar)
{
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<Distance, angle::Radians>>(0, g(3., pi/3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<Distance, angle::Radians>>(1, g(3., pi/3))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<Distance, angle::Radians>>(2, g(3., pi/3))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<angle::Radians, Distance>>(0, g(pi/3, 3.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<angle::Radians, Distance>>(1, g(pi/3, 3.))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<angle::Radians, Distance>>(2, g(pi/3, 3.))), 3., 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<Distance, angle::Degrees>>(0, g(3., 60.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<Distance, angle::Degrees>>(1, g(3., 60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Polar<Distance, angle::Degrees>>(2, g(3., 60.))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Degrees>>>(0, g(1.1, 3., 60.))), 1.1, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Degrees>>>(1, g(1.1, 3., 60.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Degrees>>>(2, g(1.1, 3., 60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Degrees>>>(3, g(1.1, 3., 60.))), std::sqrt(3)/2, 1e-6);
}


TEST(coefficients, toEuclidean_spherical)
{
  EXPECT_NEAR((internal::to_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(0, g(2., pi/6, pi/3))), 2., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(1, g(2., pi/6, pi/3))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(2, g(2., pi/6, pi/3))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(3, g(2., pi/6, pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(0, g(3., 2., pi/6, pi/3))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(1, g(3., 2., pi/6, pi/3))), 2., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(2, g(3., 2., pi/6, pi/3))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(3, g(3., 2., pi/6, pi/3))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(4, g(3., 2., pi/6, pi/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(0, g(2., pi/6, pi/3, 3.))), 2., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(1, g(2., pi/6, pi/3, 3.))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(2, g(2., pi/6, pi/3, 3.))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(3, g(2., pi/6, pi/3, 3.))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(4, g(2., pi/6, pi/3, 3.))), 3., 1e-6);
}


TEST(coefficients, toEuclidean_dynamic)
{
  EXPECT_EQ(internal::to_euclidean_coeff(DynamicCoefficients<int> {Axis {}}, 0, g(3)), 3);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients<double> {Axis {}}, 0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients<double> {Axis {}, Axis {}}, 0, g(3., 4.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients<double> {Axis {}, Axis {}}, 1, g(3., 4.))), 4., 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients<double> {Axis {}, angle::Degrees {}}, 0, g(3., 60.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients<double> {Axis {}, angle::Degrees {}}, 1, g(3., 60.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients<double> {Axis {}, angle::Degrees {}}, 2, g(3., 60.))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Axis> {}}, 0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Axis, Axis> {}}, 0, g(3., 2.))), 3., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Axis, Axis> {}}, 1, g(3., 2.))), 2., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<angle::PositiveRadians, Axis> {}}, 0, g(pi/3, 2.))), 0.5, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<angle::PositiveRadians, Axis> {}}, 1, g(pi/3, 2.))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<angle::PositiveRadians, Axis> {}}, 2, g(pi/3, 2.))), 2., 1e-6);

  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis> {}}, 0, g(2., pi/6, pi/3, 3.))), 2., 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis> {}}, 1, g(2., pi/6, pi/3, 3.))), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis> {}}, 2, g(2., pi/6, pi/3, 3.))), 0.25, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis> {}}, 3, g(2., pi/6, pi/3, 3.))), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((internal::to_euclidean_coeff(DynamicCoefficients {Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis> {}}, 4, g(2., pi/6, pi/3, 3.))), 3., 1e-6);
}


TEST(coefficients, fromEuclidean_axis_angle)
{
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis>>(0, g(3.))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis>>(0, g(-3.))), -3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<angle::Radians>>(0, g(0.5, std::sqrt(3) / 2))), pi / 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<angle::Degrees>>(0, g(0.5, std::sqrt(3) / 2))), 60, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, angle::Radians>>(0, g(3, 0.5, -std::sqrt(3) / 2))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, angle::Radians>>(1, g(3, 0.5, -std::sqrt(3) / 2))), -pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<angle::Radians, Axis>>(0, g(0.5, std::sqrt(3) / 2, 3))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<angle::Radians, Axis>>(1, g(0.5, std::sqrt(3) / 2, 3))), 3, 1e-6);
}


TEST(coefficients, fromEuclidean_distance_inclination)
{
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Distance>>(0, g(3.))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Distance>>(0, g(-3.))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<inclination::Radians>>(0, g(0.5, std::sqrt(3) / 2))), pi / 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<inclination::Degrees>>(0, g(0.5, std::sqrt(3) / 2))), 60, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Distance, inclination::Radians>>(0, g(3, 0.5, -std::sqrt(3) / 2))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Distance, inclination::Radians>>(1, g(3, 0.5, -std::sqrt(3) / 2))), -pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<inclination::Radians, Distance>>(0, g(0.5, std::sqrt(3) / 2, 3))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<inclination::Radians, Distance>>(1, g(0.5, std::sqrt(3) / 2, 3))), 3, 1e-6);
}


TEST(coefficients, fromEuclidean_polar)
{
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Polar<Distance, angle::Radians>>>(0, g(3., 0.5, std::sqrt(3) / 2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Polar<Distance, angle::Radians>>>(1, g(3., 0.5, std::sqrt(3) / 2))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Radians>>>(0, g(1.1, 3, 0.5, -std::sqrt(3) / 2))), 1.1, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Radians>>>(1, g(1.1, 3, 0.5, -std::sqrt(3) / 2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Polar<Distance, angle::Radians>>>(2, g(1.1, 3, 0.5, -std::sqrt(3) / 2))), -pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(0, g(0.5, std::sqrt(3)/2, 3, 1.1))), 60, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(1, g(0.5, std::sqrt(3)/2, 3, 1.1))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(2, g(0.5, std::sqrt(3)/2, 3, 1.1))), 1.1, 1e-6);
}


TEST(coefficients, fromEuclidean_spherical)
{
  EXPECT_NEAR((internal::from_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(0,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(1,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Spherical<Distance, angle::Radians, inclination::Radians>>(2,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/3, 1e-6);

  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(0,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(1,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(2,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(3,
    g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/3, 1e-6);

  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(0,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), 2., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(1,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(2,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), pi/3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff<Coefficients<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>>(3,
    g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.))), 3., 1e-6);
}


TEST(coefficients, fromEuclidean_dynamic)
{
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients<int> {Axis {}}, 0, g(3))), 3, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Axis {}}, 0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Axis {}, Axis {}}, 0, g(3., 2.))), 3., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Axis {}, Axis {}}, 1, g(3., 2.))), 2., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Coefficients<Axis> {}}, 0, g(3.))), 3., 1e-6);

  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    0, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 3., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    1, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), 2., 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    2, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/6, 1e-6);
  EXPECT_NEAR((internal::from_euclidean_coeff(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    3, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2))), pi/3, 1e-6);
}


TEST(coefficients, wrap_get_axis_angle)
{
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


TEST(coefficients, wrap_get_distance_inclination)
{
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


TEST(coefficients, wrap_get_polar)
{
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(0, g(3., -2., 1.1*pi))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(1, g(3., -2., 1.1*pi))), 2., 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(2, g(3., 2., 1.1*pi))), -0.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<Distance, angle::Radians>>>(2, g(3., -2., 1.1*pi))), 0.1*pi, 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(0, g(190., 2., 4.))), -170, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(0, g(190., -2., 4.))), 10, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(1, g(190., -2., 4.))), 2, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<angle::Degrees, Distance>, Axis>>(2, g(190., -2., -4.))), -4, 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>>(0, g(-2., -0.1*pi, 4.))), 2, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>>(1, g(2., -0.1*pi, 4.))), 1.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>>(1, g(-2., -0.1*pi, 4.))), 0.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Polar<Distance, angle::PositiveRadians>, Axis>>(2, g(-2., -0.1*pi, -4.))), -4, 1e-6);

  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>>(0, g(5., -10., 2.))), 5, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>>(1, g(5., -10., 2.))), 350, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>>(1, g(5., -10., -2.))), 170, 1e-6);
  EXPECT_NEAR((internal::wrap_get<Coefficients<Axis, Polar<angle::PositiveDegrees, Distance>>>(2, g(5., -10., -2.))), 2, 1e-6);
}


TEST(coefficients, wrap_get_spherical)
{
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


TEST(coefficients, wrap_get_dynamic)
{
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients<int> {Axis {}}, 0, g(3))), 3, 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Axis {}}, 0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Axis {}, Axis {}}, 0, g(3., 2.))), 3., 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Axis {}, Axis {}}, 1, g(3., 2.))), 2., 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Coefficients<Axis> {}}, 0, g(3.))), 3., 1e-6);

  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Coefficients<Polar<Distance, angle::PositiveRadians>, Axis> {}},
    0, g(-2., -0.1*pi, 4.))), 2, 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Coefficients<Polar<Distance, angle::PositiveRadians>, Axis> {}},
    1, g(2., -0.1*pi, 4.))), 1.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Coefficients<Polar<Distance, angle::PositiveRadians>, Axis> {}},
    1, g(-2., -0.1*pi, 4.))), 0.9*pi, 1e-6);
  EXPECT_NEAR((internal::wrap_get(DynamicCoefficients {Coefficients<Polar<Distance, angle::PositiveRadians>, Axis> {}},
    2, g(-2., -0.1*pi, -4.))), -4, 1e-6);
}


TEST(coefficients, wrap_set_axis_angle)
{
  std::array<double, 2> a1 = {0, 0};
  auto set_coeff = [&](const std::size_t i, double s) {a1[i] = s; };
  auto get_coeff = [&](const std::size_t i) {return a1[i]; };

  internal::wrap_set<Coefficients<Axis, Axis>>(1, 2.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  internal::wrap_set<Coefficients<Axis, angle::Radians>>(1, 1.2*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  internal::wrap_set<Coefficients<angle::PositiveRadians, Axis>>(0, -0.2*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  internal::wrap_set<Coefficients<angle::Degrees, Axis>>(0, 200., set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  internal::wrap_set<Coefficients<Axis, angle::PositiveDegrees>>(1, -20., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  internal::wrap_set<Coefficients<Axis, angle::Circle>>(1, -0.2, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);
}


TEST(coefficients, wrap_set_distance_inclination)
{
  std::array<double, 2> a1 = {0, 0};
  auto set_coeff = [&](const std::size_t i, double s) {a1[i] = s; };
  auto get_coeff = [&](const std::size_t i) {return a1[i]; };

  internal::wrap_set<Coefficients<Distance, Distance>>(0, 2.1, set_coeff, get_coeff);
  internal::wrap_set<Coefficients<Distance, Distance>>(1, -2.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  internal::wrap_set<Coefficients<inclination::Radians, inclination::Degrees>>(0, 0.7*pi, set_coeff, get_coeff);
  internal::wrap_set<Coefficients<inclination::Radians, inclination::Degrees>>(1, 110., set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  internal::wrap_set<Coefficients<inclination::Degrees, inclination::Radians>>(0, 200., set_coeff, get_coeff);
  internal::wrap_set<Coefficients<inclination::Degrees, inclination::Radians>>(1, 1.2*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
}


TEST(coefficients, wrap_set_polar)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto set_coeff = [&](const std::size_t i, double s) {a1[i] = s; };
  auto get_coeff = [&](const std::size_t i) {return a1[i]; };

  internal::wrap_set<Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>>(0, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  internal::wrap_set<Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>>(1, 1.1*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], -0.9*pi, 1e-6);
  internal::wrap_set<Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>>(2, 190., set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  internal::wrap_set<Coefficients<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>>(3, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};
  internal::wrap_set<Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>>(0, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  internal::wrap_set<Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>>(1, -0.1*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 1.9*pi, 1e-6);
  internal::wrap_set<Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>>(2, -10., set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  internal::wrap_set<Coefficients<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>>(3, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(coefficients, wrap_set_spherical)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto set_coeff = [&](const std::size_t i, double s) {a1[i] = s; };
  auto get_coeff = [&](const std::size_t i) {return a1[i]; };

  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(0, -3.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(1, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(2, 1.1*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(3, 0.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(1, -3., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(3, -0.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>>(3, -2.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>>(0, -3.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>>(1, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>>(2, 0.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>>(3, 1.1*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>>(1, -3., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>>(2, 3.4*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};
  internal::wrap_set<Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>>(0, -3.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>>(2, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>>(1, -170., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>>(3, 100., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>>(1, 550., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  internal::wrap_set<Coefficients<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>>(2, -3., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}


TEST(coefficients, wrap_set_dynamic)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto set_coeff = [&](const std::size_t i, double s) {a1[i] = s; };
  auto get_coeff = [&](const std::size_t i) {return a1[i]; };

  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Axis, Axis, Axis> {}},
    1, 2.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, angle::Radians, Axis, Axis> {}},
      1, 1.2*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<angle::PositiveRadians, Axis, Axis, Axis> {}},
    0, -0.2*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    0, -3.1, set_coeff, get_coeff);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    1, -2., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    2, 1.1*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    3, 0.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    1, -3., set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    3, -0.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  internal::wrap_set(DynamicCoefficients {Coefficients<Axis, Spherical<Distance, angle::Radians, inclination::Radians>> {}},
    3, -2.6*pi, set_coeff, get_coeff);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
}
