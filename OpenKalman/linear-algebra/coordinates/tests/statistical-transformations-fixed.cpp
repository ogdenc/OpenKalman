/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for statistical coordinate functions
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp"
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

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


TEST(coordinates, toEuclidean_axis_angle_fixed)
{
  EXPECT_NEAR((to_stat_space(Axis{}, g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Radians{}, g(pi/3), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Radians{}, g(pi/3), 1u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Degrees{}, g(60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Degrees{}, g(30.), 1u)), 0.5, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Axis>{}, g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Axis>{}, g(3., 2.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Axis>{}, g(3., 2.), 1u)), 2., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Dimensions<2>, Dimensions<3>>{}, g(3., 2., 5., 6., 7.), 4u)), 7., 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<angle::Degrees>{}, g(-60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<angle::Degrees>{}, g(-30.), 1u)), -0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<angle::Radians, angle::PositiveRadians>{}, g(pi / 3, pi / 6), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<angle::Radians, angle::PositiveRadians>{}, g(pi / 3, pi / 6), 1u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<angle::Radians, angle::Radians>{}, g(pi / 3, pi / 6), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<angle::Radians, angle::Radians>{}, g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<angle::Radians, Dimensions<2>, angle::Radians>{}, g(pi / 3, 3., 4., pi / 6), 5u)), 0.5, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Axis, angle::Radians>{}, g(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, angle::Radians>{}, g(3., pi / 3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, angle::Radians>{}, g(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);
}


TEST(coordinates, toEuclidean_distance_inclination_fixed)
{
  EXPECT_NEAR((to_stat_space(Distance{}, g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Distance{}, g(-3.), 0u)), -3., 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Radians{}, g(pi/3), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Radians{}, g(pi/3), 1u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Degrees{}, g(60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(angle::Degrees{}, g(30.), 1u)), 0.5, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Distance>{}, g(-3.), 0u)), -3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Distance, Distance>{}, g(3., -2.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Distance, Distance>{}, g(3., -2.), 1u)), -2., 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<inclination::Degrees>{}, g(-60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<inclination::Degrees>{}, g(-30.), 1u)), -0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 1u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<inclination::Radians, inclination::Radians>{}, g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Distance, inclination::Radians>{}, g(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Distance, inclination::Radians>{}, g(3., pi / 3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Distance, inclination::Radians>{}, g(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);
}


TEST(coordinates, toEuclidean_polar_fixed)
{
  EXPECT_NEAR((to_stat_space(Polar<Distance, angle::Radians>{}, g(3., pi/3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<Distance, angle::Radians>{}, g(3., pi/3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<Distance, angle::Radians>{}, g(3., pi/3), 2u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<angle::Radians, Distance>{}, g(pi/3, 3.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<angle::Radians, Distance>{}, g(pi/3, 3.), 1u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<angle::Radians, Distance>{}, g(pi/3, 3.), 2u)), 3., 1e-6);

  EXPECT_NEAR((to_stat_space(Polar<Distance, angle::Degrees>{}, g(3., 60.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<Distance, angle::Degrees>{}, g(3., 60.), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(Polar<Distance, angle::Degrees>{}, g(3., 60.), 2u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 0u)), 1.1, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 1u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 2u)), 0.5, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 3u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Polar<Distance, angle::Degrees>>{}, g(1.1, 3., 60.), 3u)), std::sqrt(3)/2, 1e-6);
}


TEST(coordinates, toEuclidean_spherical_fixed)
{
  EXPECT_NEAR((to_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 0u)), 2., 1e-6);
  EXPECT_NEAR((to_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 1u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 2u)), 0.25, 1e-6);
  EXPECT_NEAR((to_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., pi/6, pi/3), 3u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 1u)), 2., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 2u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 3u)), 0.25, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., pi/6, pi/3), 4u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 0u)), 2., 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 1u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 2u)), 0.25, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 3u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., pi/6, pi/3, 3.), 4u)), 3., 1e-6);
}


TEST(coordinates, fromEuclidean_axis_angle_fixed)
{
  EXPECT_NEAR((from_stat_space(std::tuple<Axis>{}, g(3.), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis>{}, g(-3.), 0u)), -3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<angle::Radians>{}, g(0.5, std::sqrt(3) / 2), 0u)), pi / 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<angle::Degrees>{}, g(0.5, std::sqrt(3) / 2), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, angle::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, angle::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 1u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<angle::Radians, Axis>{}, g(0.5, std::sqrt(3) / 2, 3), 0u)), pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<angle::Radians, Axis>{}, g(0.5, std::sqrt(3) / 2, 3), 1u)), 3, 1e-6);
}


TEST(coordinates, fromEuclidean_distance_inclination_fixed)
{
  EXPECT_NEAR((from_stat_space(std::tuple<Distance>{}, g(3.), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Distance>{}, g(-3.), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<inclination::Radians>{}, g(0.5, std::sqrt(3) / 2), 0u)), pi / 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<inclination::Degrees>{}, g(0.5, std::sqrt(3) / 2), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Distance, inclination::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Distance, inclination::Radians>{}, g(3, 0.5, -std::sqrt(3) / 2), 1u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<inclination::Radians, Distance>{}, g(0.5, std::sqrt(3) / 2, 3), 0u)), pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<inclination::Radians, Distance>{}, g(0.5, std::sqrt(3) / 2, 3), 1u)), 3, 1e-6);
}


TEST(coordinates, fromEuclidean_polar_fixed)
{
  EXPECT_NEAR((from_stat_space(std::tuple<Polar<Distance, angle::Radians>>{}, g(3., 0.5, std::sqrt(3) / 2), 0u)), 3., 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Polar<Distance, angle::Radians>>{}, g(3., 0.5, std::sqrt(3) / 2), 1u)), pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0u)), 1.1, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1u)), 3., 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(0.5, std::sqrt(3)/2, 3, 1.1), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(0.5, std::sqrt(3)/2, 3, 1.1), 1u)), 3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(0.5, std::sqrt(3)/2, 3, 1.1), 2u)), 1.1, 1e-6);
}


TEST(coordinates, fromEuclidean_spherical_fixed)
{
  EXPECT_NEAR((from_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0u)), 2., 1e-6);
  EXPECT_NEAR((from_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1u)), pi/6, 1e-6);
  EXPECT_NEAR((from_stat_space(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2u)), pi/3, 1e-6);

  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0u)), 3., 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1u)), 2., 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2u)), pi/6, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3u)), pi/3, 1e-6);

  EXPECT_NEAR((from_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0u)), 2., 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1u)), pi/6, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2u)), pi/3, 1e-6);
  EXPECT_NEAR((from_stat_space(std::tuple<Spherical<Distance, angle::Radians, inclination::Radians>, Axis>{}, g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3u)), 3., 1e-6);
}


TEST(coordinates, get_wrapped_component_axis_angle_fixed)
{
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis>{}, g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis>{}, g(-3.), 0u)), -3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians>{}, g(1.2*pi), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::PositiveRadians>{}, g(-0.2*pi), 0u)), 1.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Degrees>{}, g(200.), 0u)), -160., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::PositiveDegrees>{}, g(-20.), 0u)), 340., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Circle>{}, g(-0.2), 0u)), 0.8, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, angle::Radians>{}, g(3, 1.2*pi), 0u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, angle::Radians>{}, g(3, 1.2*pi), 1u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians, Axis>{}, g(1.2*pi, 3), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians, Axis>{}, g(1.2*pi, 3), 1u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians, Axis, angle::Radians, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians, Axis, angle::Radians, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 1u)), 5, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians, Axis, angle::Radians, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 2u)), -0.7*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<angle::Radians, Axis, angle::Radians, Axis>{}, g(1.2*pi, 5, 1.3*pi, 3), 3u)), 3, 1e-6);
}


TEST(coordinates, get_wrapped_component_distance_inclination_fixed)
{
  EXPECT_NEAR((get_wrapped_component(std::tuple<Distance>{}, g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Distance>{}, g(-3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<inclination::Radians>{}, g(0.7*pi), 0u)), 0.3*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<inclination::Radians>{}, g(1.2*pi), 0u)), -0.2*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<inclination::Degrees>{}, g(110.), 0u)), 70., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<inclination::Degrees>{}, g(200.), 0u)), -20., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Distance, inclination::Radians>{}, g(3, 0.7*pi), 0u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Distance, inclination::Radians>{}, g(3, 0.7*pi), 1u)), 0.3*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<inclination::Radians, Distance>{}, g(1.2*pi, 3), 0u)), -0.2*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<inclination::Radians, Distance>{}, g(1.2*pi, 3), 1u)), 3, 1e-6);
}


TEST(coordinates, get_wrapped_component_polar_fixed)
{
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(3., -2., 1.1*pi), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(3., -2., 1.1*pi), 1u)), 2., 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(3., 2., 1.1*pi), 2u)), -0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<Distance, angle::Radians>>{}, g(3., -2., 1.1*pi), 2u)), 0.1*pi, 1e-6);

  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(190., 2., 4.), 0u)), -170, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(190., -2., 4.), 0u)), 10, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(190., -2., 4.), 1u)), 2, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<angle::Degrees, Distance>, Axis>{}, g(190., -2., -4.), 2u)), -4, 1e-6);

  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(-2., -0.1*pi, 4.), 0u)), 2, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(2., -0.1*pi, 4.), 1u)), 1.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(-2., -0.1*pi, 4.), 1u)), 0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Axis>{}, g(-2., -0.1*pi, -4.), 2u)), -4, 1e-6);

  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., 2.), 0u)), 5, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., 2.), 1u)), 350, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., -2.), 1u)), 170, 1e-6);
  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Polar<angle::PositiveDegrees, Distance>>{}, g(5., -10., -2.), 2u)), 2, 1e-6);
}


TEST(coordinates, get_wrapped_component_spherical_fixed)
{
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 1.1*pi, 0.6*pi), 0u)), 2., 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 0.5*pi, 0.6*pi), 1u)), -0.5*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 0.5*pi, 0.6*pi), 2u)), 0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 0.5*pi, 0.6*pi), 1u)), 0.5*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 0.5*pi, 0.6*pi), 2u)), -0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 1.1*pi, 0.6*pi), 1u)), 0.1*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(2., 1.1*pi, 0.6*pi), 2u)), 0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 1.1*pi, 0.6*pi), 1u)), -0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(Spherical<Distance, angle::Radians, inclination::Radians>{}, g(-2., 1.1*pi, 0.6*pi), 2u)), -0.4*pi, 1e-6);

  EXPECT_NEAR((get_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, g(3., -2., 1.1*pi, 0.6*pi), 3u)), -0.4*pi, 1e-6);
}


TEST(coordinates, set_wrapped_component_axis_angle_fixed)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) { return a1[i]; }};

  set_wrapped_component(std::tuple<Axis, Axis>{}, s, g, 2.1, 1u);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  set_wrapped_component(std::tuple<Axis, angle::Radians>{}, s, g, 1.2*pi, 1u);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  set_wrapped_component(std::tuple<angle::PositiveRadians, Axis>{}, s, g, -0.2*pi, 0u);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  set_wrapped_component(std::tuple<angle::Degrees, Axis>{}, s, g, 200., 0u);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  set_wrapped_component(std::tuple<Axis, angle::PositiveDegrees>{}, s, g, -20., 1u);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  set_wrapped_component(std::tuple<Axis, angle::Circle>{}, s, g, -0.2, 1u);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);
}


TEST(coordinates, set_wrapped_component_distance_inclination_fixed)
{
  std::array<float, 2> a1 = {0, 0};
  auto s = std::function {[&](const float& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  set_wrapped_component(std::tuple<Distance, Distance>{}, s, g, 2.1, 0u);
  set_wrapped_component(std::tuple<Distance, Distance>{}, s, g, -2.1, 1u);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  set_wrapped_component(std::tuple<inclination::Radians, inclination::Degrees>{}, s, g, 0.7*pi, 0u);
  set_wrapped_component(std::tuple<inclination::Radians, inclination::Degrees>{}, s, g, 110., 1u);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  set_wrapped_component(std::tuple<inclination::Degrees, inclination::Radians>{}, s, g, 200., 0u);
  set_wrapped_component(std::tuple<inclination::Degrees, inclination::Radians>{}, s, g, 1.2*pi, 1u);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
}


TEST(coordinates, set_wrapped_component_polar_fixed)
{
  std::array<double, 4> a1 = {1, 0.8*pi, 0, 1};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  set_wrapped_component(std::tuple<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, -2., 0u);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
  set_wrapped_component(std::tuple<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, 1.1*pi, 1u);
  EXPECT_NEAR(a1[1], -0.9*pi, 1e-6);
  set_wrapped_component(std::tuple<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, 190., 2u);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  set_wrapped_component(std::tuple<Polar<Distance, angle::Radians>, Polar<angle::Degrees, Distance>>{}, s, g, -2., 3u);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -2., 0u);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  set_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -0.1*pi, 1u);
  EXPECT_NEAR(a1[1], 1.9*pi, 1e-6);
  set_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -10., 2u);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  set_wrapped_component(std::tuple<Polar<Distance, angle::PositiveRadians>, Polar<angle::PositiveDegrees, Distance>>{}, s, g, -2., 3u);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(coordinates, set_wrapped_component_spherical_fixed)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -2., 1u);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, 1.1*pi, 2u);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, 0.6*pi, 3u);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -3., 1u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -0.6*pi, 3u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, angle::Radians, inclination::Radians>>{}, s, g, -2.6*pi, 3u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(std::tuple<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, -2., 1u);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, 0.6*pi, 2u);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, 1.1*pi, 3u);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, -3., 1u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<Distance, inclination::Radians, angle::Radians>>{}, s, g, 3.4*pi, 2u);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(std::tuple<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -2., 2u);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -170., 1u);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, 100., 3u);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, 550., 1u);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  set_wrapped_component(std::tuple<Axis, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>{}, s, g, -3., 2u);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}


