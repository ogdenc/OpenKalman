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

using D = DynamicTypedIndex<double>;


TEST(index_descriptors, toEuclidean_axis_angle_fixed)
{
  EXPECT_NEAR((Axis::to_euclidean_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((TypedIndex<Axis>::to_euclidean_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Axis>::to_euclidean_element<double>(g(3., 2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Axis>::to_euclidean_element<double>(g(3., 2.), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((TypedIndex<Dimensions<2>, Dimensions<3>>::to_euclidean_element<double>(g(3., 2., 5., 6., 7.), 4, 0)), 7., 1e-6);

  EXPECT_NEAR((TypedIndex<angle::Degrees>::to_euclidean_element<double>(g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Degrees>::to_euclidean_element<double>(g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, angle::PositiveRadians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, angle::PositiveRadians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, angle::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, angle::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, Dimensions<2>, angle::Radians>::to_euclidean_element<double>(g(pi / 3, 3., 4., pi / 6), 5, 0)), 0.5, 1e-6);

  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::to_euclidean_element<double>(g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::to_euclidean_element<double>(g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::to_euclidean_element<double>(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_axis_angle_dynamic)
{
  EXPECT_NEAR((D {Axis{}}.to_euclidean_element(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {angle::Radians{}}.to_euclidean_element(g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {angle::Radians{}}.to_euclidean_element(g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {angle::Degrees{}}.to_euclidean_element(g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {angle::Degrees{}}.to_euclidean_element(g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis>{}}.to_euclidean_element(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Axis>{}}.to_euclidean_element(g(3., 2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Axis>{}}.to_euclidean_element(g(3., 2.), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((D {Axis{}, D {Axis{}}}.to_euclidean_element(g(3., 2.), 1, 0)), 2., 1e-6);

  EXPECT_NEAR((D {TypedIndex<angle::Degrees>{}}.to_euclidean_element(g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Degrees>{}}.to_euclidean_element(g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, angle::PositiveRadians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, angle::PositiveRadians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, angle::Radians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, angle::Radians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {D {angle::Radians{}}, angle::Radians{}}.to_euclidean_element(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {angle::Radians{}, D {angle::Radians{}}}.to_euclidean_element(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {angle::Radians{}, Dimensions<2>{}, D {angle::Radians{}}}.to_euclidean_element(g(pi / 3, 3., 4., pi / 6), 5, 0)), 0.5, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.to_euclidean_element(g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.to_euclidean_element(g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.to_euclidean_element(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {Axis{}, angle::Radians{}}.to_euclidean_element(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_distance_inclination_fixed)
{
  EXPECT_NEAR((Distance::to_euclidean_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((Distance::to_euclidean_element<double>(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Radians::to_euclidean_element<double>(g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((angle::Degrees::to_euclidean_element<double>(g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((TypedIndex<Distance>::to_euclidean_element<double>(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, Distance>::to_euclidean_element<double>(g(3., -2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, Distance>::to_euclidean_element<double>(g(3., -2.), 1, 0)), -2., 1e-6);

  EXPECT_NEAR((TypedIndex<inclination::Degrees>::to_euclidean_element<double>(g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Degrees>::to_euclidean_element<double>(g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, inclination::Radians>::to_euclidean_element<double>(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);

  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::to_euclidean_element<double>(g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::to_euclidean_element<double>(g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::to_euclidean_element<double>(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
}


TEST(index_descriptors, toEuclidean_distance_inclination_dynamic)
{
  EXPECT_NEAR((D {Distance{}}.to_euclidean_element(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {Distance{}}.to_euclidean_element(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((D {angle::Radians{}}.to_euclidean_element(g(pi/3), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {angle::Radians{}}.to_euclidean_element(g(pi/3), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {angle::Degrees{}}.to_euclidean_element(g(60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {angle::Degrees{}}.to_euclidean_element(g(30.), 1, 0)), 0.5, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Distance>{}}.to_euclidean_element(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, Distance>{}}.to_euclidean_element(g(3., -2.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, Distance>{}}.to_euclidean_element(g(3., -2.), 1, 0)), -2., 1e-6);
  EXPECT_NEAR((D {Distance{}, D{Distance{}}}.to_euclidean_element(g(3., -2.), 1, 0)), -2., 1e-6);

  EXPECT_NEAR((D {TypedIndex<inclination::Degrees>{}}.to_euclidean_element(g(-60.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Degrees>{}}.to_euclidean_element(g(-30.), 1, 0)), -0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, inclination::Radians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, inclination::Radians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 1, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, inclination::Radians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, inclination::Radians>{}}.to_euclidean_element(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {D{inclination::Radians{}}, inclination::Radians{}}.to_euclidean_element(g(pi / 3, pi / 6), 3, 0)), 0.5, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.to_euclidean_element(g(3., pi / 3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.to_euclidean_element(g(3., pi / 3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.to_euclidean_element(g(3., pi / 3), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {Distance{}, D{inclination::Radians{}, Distance{}}}.to_euclidean_element(g(3., pi / 3, 4), 2, 0)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((D {Distance{}, D{inclination::Radians{}, Distance{}}}.to_euclidean_element(g(3., pi / 3, 4), 3, 0)), 4, 1e-6);
}


TEST(index_descriptors, toEuclidean_polar_fixed)
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

  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 2, 0)), 0.5, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Degrees>>::to_euclidean_element<double>(g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
}


TEST(index_descriptors, toEuclidean_polar_dynamic)
{
  EXPECT_NEAR((D {Polar<Distance, angle::Radians>{}}.to_euclidean_element(g(3., pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {Polar<Distance, angle::Radians>{}}.to_euclidean_element(g(3., pi/3), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {Polar<Distance, angle::Radians>{}}.to_euclidean_element(g(3., pi/3), 2, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {Polar<angle::Radians, Distance>{}}.to_euclidean_element(g(pi/3, 3.), 0, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {Polar<angle::Radians, Distance>{}}.to_euclidean_element(g(pi/3, 3.), 1, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {Polar<angle::Radians, Distance>{}}.to_euclidean_element(g(pi/3, 3.), 2, 0)), 3., 1e-6);

  EXPECT_NEAR((D {Polar<Distance, angle::Degrees>{}}.to_euclidean_element(g(3., 60.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {Polar<Distance, angle::Degrees>{}}.to_euclidean_element(g(3., 60.), 1, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {Polar<Distance, angle::Degrees>{}}.to_euclidean_element(g(3., 60.), 2, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}.to_euclidean_element(g(1.1, 3., 60.), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}.to_euclidean_element(g(1.1, 3., 60.), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}.to_euclidean_element(g(1.1, 3., 60.), 2, 0)), 0.5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Degrees>>{}}.to_euclidean_element(g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {Axis{}, D{Polar<Distance, angle::Degrees>{}}}.to_euclidean_element(g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {D{Axis{}}, Polar<Distance, angle::Degrees>{}}.to_euclidean_element(g(1.1, 3., 60.), 3, 0)), std::sqrt(3)/2, 1e-6);
}


TEST(index_descriptors, toEuclidean_spherical_fixed)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::to_euclidean_element<double>(g(2., pi/6, pi/3), 3, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 2, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 3, 0)), 0.25, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::to_euclidean_element<double>(g(3., 2., pi/6, pi/3), 4, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::to_euclidean_element<double>(g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
}


TEST(index_descriptors, toEuclidean_spherical_dynamic)
{
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(g(2., pi/6, pi/3), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(g(2., pi/6, pi/3), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(g(2., pi/6, pi/3), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.to_euclidean_element(g(2., pi/6, pi/3), 3, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.to_euclidean_element(g(3., 2., pi/6, pi/3), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.to_euclidean_element(g(3., 2., pi/6, pi/3), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.to_euclidean_element(g(3., 2., pi/6, pi/3), 2, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.to_euclidean_element(g(3., 2., pi/6, pi/3), 3, 0)), 0.25, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.to_euclidean_element(g(3., 2., pi/6, pi/3), 4, 0)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 1, 0)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 2, 0)), 0.25, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 3, 0)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
  EXPECT_NEAR((D {D{Spherical<Distance, angle::Radians, inclination::Radians>{}}, Axis{}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}, D{Axis{}}}.to_euclidean_element(g(2., pi/6, pi/3, 3.), 4, 0)), 3., 1e-6);
}


TEST(index_descriptors, fromEuclidean_axis_angle_fixed)
{
  EXPECT_NEAR((TypedIndex<Axis>::from_euclidean_element<double>(g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis>::from_euclidean_element<double>(g(-3.), 0, 0)), -3, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Degrees>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_axis_angle_dynamic)
{
  EXPECT_NEAR((D {TypedIndex<Axis>{}}.from_euclidean_element(g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis>{}}.from_euclidean_element(g(-3.), 0, 0)), -3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Degrees>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.from_euclidean_element(g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.from_euclidean_element(g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, Axis>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, Axis>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_distance_inclination_fixed)
{
  EXPECT_NEAR((TypedIndex<Distance>::from_euclidean_element<double>(g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Distance>::from_euclidean_element<double>(g(-3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Degrees>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::from_euclidean_element<double>(g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, Distance>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, Distance>::from_euclidean_element<double>(g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_distance_inclination_dynamic)
{
  EXPECT_NEAR((D {TypedIndex<Distance>{}}.from_euclidean_element(g(3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance>{}}.from_euclidean_element(g(-3.), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2), 0, 0)), pi / 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Degrees>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.from_euclidean_element(g(3, 0.5, -std::sqrt(3) / 2), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.from_euclidean_element(g(3, 0.5, -std::sqrt(3) / 2), 1, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, Distance>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2, 3), 0, 0)), pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, Distance>{}}.from_euclidean_element(g(0.5, std::sqrt(3) / 2, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, fromEuclidean_polar_fixed)
{
  EXPECT_NEAR((TypedIndex<Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(3., 0.5, std::sqrt(3) / 2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(3., 0.5, std::sqrt(3) / 2), 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::from_euclidean_element<double>(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3)/2, 3, 1.1), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::from_euclidean_element<double>(g(0.5, std::sqrt(3)/2, 3, 1.1), 2, 0)), 1.1, 1e-6);
}


TEST(index_descriptors, fromEuclidean_polar_dynamic)
{
  EXPECT_NEAR((D {TypedIndex<Polar<Distance, angle::Radians>>{}}.from_euclidean_element(g(3., 0.5, std::sqrt(3) / 2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<Distance, angle::Radians>>{}}.from_euclidean_element(g(3., 0.5, std::sqrt(3) / 2), 1, 0)), pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.from_euclidean_element(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0, 0)), 1.1, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.from_euclidean_element(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.from_euclidean_element(g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2, 0)), -pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.from_euclidean_element(g(0.5, std::sqrt(3)/2, 3, 1.1), 0, 0)), 60, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.from_euclidean_element(g(0.5, std::sqrt(3)/2, 3, 1.1), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.from_euclidean_element(g(0.5, std::sqrt(3)/2, 3, 1.1), 2, 0)), 1.1, 1e-6);
}


TEST(index_descriptors, fromEuclidean_spherical_fixed)
{
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((Spherical<Distance, angle::Radians, inclination::Radians>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/3, 1e-6);

  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/6, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::from_euclidean_element<double>(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3, 0)), pi/3, 1e-6);

  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2, 0)), pi/3, 1e-6);
  EXPECT_NEAR((TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>::from_euclidean_element<double>(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
}


TEST(index_descriptors, fromEuclidean_spherical_dynamic)
{
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/3, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.from_euclidean_element(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.from_euclidean_element(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.from_euclidean_element(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2, 0)), pi/6, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.from_euclidean_element(g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3, 0)), pi/3, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1, 0)), pi/6, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2, 0)), pi/3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
  EXPECT_NEAR((D {D{Spherical<Distance, angle::Radians, inclination::Radians>{}}, Axis{}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}, D{Axis{}}}.from_euclidean_element(g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3, 0)), 3., 1e-6);
}


TEST(index_descriptors, wrap_get_element_axis_angle_fixed)
{
  EXPECT_NEAR((TypedIndex<Axis>::wrap_get_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis>::wrap_get_element<double>(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians>::wrap_get_element<double>(g(1.2*pi), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::PositiveRadians>::wrap_get_element<double>(g(-0.2*pi), 0, 0)), 1.8*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Degrees>::wrap_get_element<double>(g(200.), 0, 0)), -160., 1e-6);
  EXPECT_NEAR((TypedIndex<angle::PositiveDegrees>::wrap_get_element<double>(g(-20.), 0, 0)), 340., 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Circle>::wrap_get_element<double>(g(-0.2), 0, 0)), 0.8, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::wrap_get_element<double>(g(3, 1.2*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, angle::Radians>::wrap_get_element<double>(g(3, 1.2*pi), 1, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, Axis>::wrap_get_element<double>(g(1.2*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, Axis>::wrap_get_element<double>(g(1.2*pi, 3), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>::wrap_get_element<double>(g(1.2*pi, 5, 1.3*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>::wrap_get_element<double>(g(1.2*pi, 5, 1.3*pi, 3), 1, 0)), 5, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>::wrap_get_element<double>(g(1.2*pi, 5, 1.3*pi, 3), 2, 0)), -0.7*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>::wrap_get_element<double>(g(1.2*pi, 5, 1.3*pi, 3), 3, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_axis_angle_dynamic)
{
  EXPECT_NEAR((D {TypedIndex<Axis>{}}.wrap_get_element(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis>{}}.wrap_get_element(g(-3.), 0, 0)), -3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians>{}}.wrap_get_element(g(1.2*pi), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::PositiveRadians>{}}.wrap_get_element(g(-0.2*pi), 0, 0)), 1.8*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Degrees>{}}.wrap_get_element(g(200.), 0, 0)), -160., 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::PositiveDegrees>{}}.wrap_get_element(g(-20.), 0, 0)), 340., 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Circle>{}}.wrap_get_element(g(-0.2), 0, 0)), 0.8, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.wrap_get_element(g(3, 1.2*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, angle::Radians>{}}.wrap_get_element(g(3, 1.2*pi), 1, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, Axis>{}}.wrap_get_element(g(1.2*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, Axis>{}}.wrap_get_element(g(1.2*pi, 3), 1, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}.wrap_get_element(g(1.2*pi, 5, 1.3*pi, 3), 0, 0)), -0.8*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}.wrap_get_element(g(1.2*pi, 5, 1.3*pi, 3), 1, 0)), 5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}.wrap_get_element(g(1.2*pi, 5, 1.3*pi, 3), 2, 0)), -0.7*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<angle::Radians, TypedIndex<Axis, angle::Radians>, Axis>{}}.wrap_get_element(g(1.2*pi, 5, 1.3*pi, 3), 3, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_distance_inclination_fixed)
{
  EXPECT_NEAR((TypedIndex<Distance>::wrap_get_element<double>(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Distance>::wrap_get_element<double>(g(-3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians>::wrap_get_element<double>(g(0.7*pi), 0, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians>::wrap_get_element<double>(g(1.2*pi), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Degrees>::wrap_get_element<double>(g(110.), 0, 0)), 70., 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Degrees>::wrap_get_element<double>(g(200.), 0, 0)), -20., 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::wrap_get_element<double>(g(3, 0.7*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((TypedIndex<Distance, inclination::Radians>::wrap_get_element<double>(g(3, 0.7*pi), 1, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, Distance>::wrap_get_element<double>(g(1.2*pi, 3), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<inclination::Radians, Distance>::wrap_get_element<double>(g(1.2*pi, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_distance_inclination_dynamic)
{
  EXPECT_NEAR((D {TypedIndex<Distance>{}}.wrap_get_element(g(3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance>{}}.wrap_get_element(g(-3.), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians>{}}.wrap_get_element(g(0.7*pi), 0, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians>{}}.wrap_get_element(g(1.2*pi), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Degrees>{}}.wrap_get_element(g(110.), 0, 0)), 70., 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Degrees>{}}.wrap_get_element(g(200.), 0, 0)), -20., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.wrap_get_element(g(3, 0.7*pi), 0, 0)), 3, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Distance, inclination::Radians>{}}.wrap_get_element(g(3, 0.7*pi), 1, 0)), 0.3*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, Distance>{}}.wrap_get_element(g(1.2*pi, 3), 0, 0)), -0.2*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<inclination::Radians, Distance>{}}.wrap_get_element(g(1.2*pi, 3), 1, 0)), 3, 1e-6);
}


TEST(index_descriptors, wrap_get_element_polar_fixed)
{
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., 2., 1.1*pi), 2, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<Distance, angle::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi), 2, 0)), 0.1*pi, 1e-6);

  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., 2., 4.), 0, 0)), -170, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., -2., 4.), 0, 0)), 10, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., -2., 4.), 1, 0)), 2, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<angle::Degrees, Distance>, Axis>::wrap_get_element<double>(g(190., -2., -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(-2., -0.1*pi, 4.), 0, 0)), 2, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(2., -0.1*pi, 4.), 1, 0)), 1.9*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(-2., -0.1*pi, 4.), 1, 0)), 0.9*pi, 1e-6);
  EXPECT_NEAR((TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>::wrap_get_element<double>(g(-2., -0.1*pi, -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., 2.), 0, 0)), 5, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., 2.), 1, 0)), 350, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., -2.), 1, 0)), 170, 1e-6);
  EXPECT_NEAR((TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>::wrap_get_element<double>(g(5., -10., -2.), 2, 0)), 2, 1e-6);
}


TEST(index_descriptors, wrap_get_element_polar_dynamic)
{
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.wrap_get_element(g(3., -2., 1.1*pi), 0, 0)), 3., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.wrap_get_element(g(3., -2., 1.1*pi), 1, 0)), 2., 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.wrap_get_element(g(3., 2., 1.1*pi), 2, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<Distance, angle::Radians>>{}}.wrap_get_element(g(3., -2., 1.1*pi), 2, 0)), 0.1*pi, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.wrap_get_element(g(190., 2., 4.), 0, 0)), -170, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.wrap_get_element(g(190., -2., 4.), 0, 0)), 10, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.wrap_get_element(g(190., -2., 4.), 1, 0)), 2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<angle::Degrees, Distance>, Axis>{}}.wrap_get_element(g(190., -2., -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}.wrap_get_element(g(-2., -0.1*pi, 4.), 0, 0)), 2, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}.wrap_get_element(g(2., -0.1*pi, 4.), 1, 0)), 1.9*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}.wrap_get_element(g(-2., -0.1*pi, 4.), 1, 0)), 0.9*pi, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Axis>{}}.wrap_get_element(g(-2., -0.1*pi, -4.), 2, 0)), -4, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_get_element(g(5., -10., 2.), 0, 0)), 5, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_get_element(g(5., -10., 2.), 1, 0)), 350, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_get_element(g(5., -10., -2.), 1, 0)), 170, 1e-6);
  EXPECT_NEAR((D {TypedIndex<Axis, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_get_element(g(5., -10., -2.), 2, 0)), 2, 1e-6);
}


TEST(index_descriptors, wrap_get_element_spherical_fixed)
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

  EXPECT_NEAR((TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_get_element<double>(g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
}


TEST(index_descriptors, wrap_get_element_spherical_dynamic)
{
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(2., 1.1*pi, 0.6*pi), 0, 0)), 2., 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(2., 0.5*pi, 0.6*pi), 1, 0)), -0.5*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(2., 0.5*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(-2., 0.5*pi, 0.6*pi), 1, 0)), 0.5*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(-2., 0.5*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(2., 1.1*pi, 0.6*pi), 1, 0)), 0.1*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(2., 1.1*pi, 0.6*pi), 2, 0)), 0.4*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(-2., 1.1*pi, 0.6*pi), 1, 0)), -0.9*pi, 1e-6);
  EXPECT_NEAR((D {Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(-2., 1.1*pi, 0.6*pi), 2, 0)), -0.4*pi, 1e-6);

  EXPECT_NEAR((D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_get_element(g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((D {D{Axis{}}, Spherical<Distance, angle::Radians, inclination::Radians>{}}.wrap_get_element(g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
  EXPECT_NEAR((D {Axis{}, D{Spherical<Distance, angle::Radians, inclination::Radians>{}}}.wrap_get_element(g(3., -2., 1.1*pi, 0.6*pi), 3, 0)), -0.4*pi, 1e-6);
}


TEST(index_descriptors, wrap_set_element_axis_angle)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = [&](double s, const std::size_t i) { a1[i] = s; };
  auto g = [&](const std::size_t i) { return a1[i]; };

  TypedIndex<Axis, Axis>::wrap_set_element<double>(s, g, 2.1, 1, 0);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  TypedIndex<Axis, angle::Radians>::wrap_set_element<double>(s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  TypedIndex<angle::PositiveRadians, Axis>::wrap_set_element<double>(s, g, -0.2*pi, 0, 0);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  TypedIndex<angle::Degrees, Axis>::wrap_set_element<double>(s, g, 200., 0, 0);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  TypedIndex<Axis, angle::PositiveDegrees>::wrap_set_element<double>(s, g, -20., 1, 0);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  TypedIndex<Axis, angle::Circle>::wrap_set_element<double>(s, g, -0.2, 1, 0);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);

  D {TypedIndex<Axis, Axis>{}}.wrap_set_element(s, g, 2.1, 1, 0);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  D {TypedIndex<Axis, angle::Radians>{}}.wrap_set_element(s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  D {TypedIndex<angle::PositiveRadians, Axis>{}}.wrap_set_element(s, g, -0.2*pi, 0, 0);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  D {TypedIndex<angle::Degrees, Axis>{}}.wrap_set_element(s, g, 200., 0, 0);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  D {TypedIndex<Axis, angle::PositiveDegrees>{}}.wrap_set_element(s, g, -20., 1, 0);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  D {TypedIndex<Axis, angle::Circle>{}}.wrap_set_element(s, g, -0.2, 1, 0);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);
}


TEST(index_descriptors, wrap_set_element_distance_inclination)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = [&](double s, const std::size_t i) { a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  TypedIndex<Distance, Distance>::wrap_set_element<double>(s, g, 2.1, 0, 0);
  TypedIndex<Distance, Distance>::wrap_set_element<double>(s, g, -2.1, 1, 0);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  TypedIndex<inclination::Radians, inclination::Degrees>::wrap_set_element<double>(s, g, 0.7*pi, 0, 0);
  TypedIndex<inclination::Radians, inclination::Degrees>::wrap_set_element<double>(s, g, 110., 1, 0);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  TypedIndex<inclination::Degrees, inclination::Radians>::wrap_set_element<double>(s, g, 200., 0, 0);
  TypedIndex<inclination::Degrees, inclination::Radians>::wrap_set_element<double>(s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);

  D {TypedIndex<Distance, Distance>{}}.wrap_set_element(s, g, 2.1, 0, 0);
  D {TypedIndex<Distance, Distance>{}}.wrap_set_element(s, g, -2.1, 1, 0);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  D {TypedIndex<inclination::Radians, inclination::Degrees>{}}.wrap_set_element(s, g, 0.7*pi, 0, 0);
  D {TypedIndex<inclination::Radians, inclination::Degrees>{}}.wrap_set_element(s, g, 110., 1, 0);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  D {TypedIndex<inclination::Degrees, inclination::Radians>{}}.wrap_set_element(s, g, 200., 0, 0);
  D {TypedIndex<inclination::Degrees, inclination::Radians>{}}.wrap_set_element(s, g, 1.2*pi, 1, 0);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
}


TEST(index_descriptors, wrap_set_element_polar_fixed)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = [&](double s, const std::size_t i) { a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, 1.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.9*pi, 1e-6);
  TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, 190., 2, 0);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>::wrap_set_element<double>(s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -0.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], 1.9*pi, 1e-6);
  TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -10., 2, 0);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>::wrap_set_element<double>(s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(index_descriptors, wrap_set_element_polar_dynamic)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = [&](double s, const std::size_t i) { a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  D {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}.wrap_set_element(s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  D {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}.wrap_set_element(s, g, 1.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], -0.9*pi, 1e-6);
  D {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}.wrap_set_element(s, g, 190., 2, 0);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  D {TypedIndex<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}}.wrap_set_element(s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_set_element(s, g, -2., 0, 0);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_set_element(s, g, -0.1*pi, 1, 0);
  EXPECT_NEAR(a1[1], 1.9*pi, 1e-6);
  D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_set_element(s, g, -10., 2, 0);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  D {TypedIndex<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}}.wrap_set_element(s, g, -2., 3, 0);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(index_descriptors, wrap_set_element_spherical_fixed)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = [&](double s, const std::size_t i) { a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, 1.1*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, 0.6*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -0.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>::wrap_set_element<double>(s, g, -2.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, 0.6*pi, 2, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, 1.1*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>::wrap_set_element<double>(s, g, 3.4*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -2., 2, 0);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -170., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, 100., 3, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, 550., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>::wrap_set_element<double>(s, g, -3., 2, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}


TEST(index_descriptors, wrap_set_element_spherical_dynamic)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = [&](double s, const std::size_t i) { a1[i] = s; };
  auto g = [&](const std::size_t i) {return a1[i]; };

  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, 1.1*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, 0.6*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, -0.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}}.wrap_set_element(s, g, -2.6*pi, 3, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  D {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}.wrap_set_element(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}.wrap_set_element(s, g, -2., 1, 0);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}.wrap_set_element(s, g, 0.6*pi, 2, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}.wrap_set_element(s, g, 1.1*pi, 3, 0);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}.wrap_set_element(s, g, -3., 1, 0);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  D {TypedIndex<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}}.wrap_set_element(s, g, 3.4*pi, 2, 0);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  D {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}.wrap_set_element(s, g, -3.1, 0, 0);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  D {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}.wrap_set_element(s, g, -2., 2, 0);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  D {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}.wrap_set_element(s, g, -170., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  D {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}.wrap_set_element(s, g, 100., 3, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  D {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}.wrap_set_element(s, g, 550., 1, 0);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  D {TypedIndex<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}}.wrap_set_element(s, g, -3., 2, 0);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}
