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
#include "linear-algebra/coordinates/functions/to_euclidean_element.hpp"
#include "linear-algebra/coordinates/functions/from_euclidean_element.hpp"
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp"
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"
#include "linear-algebra/coordinates/functions/make_pattern_vector.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinate;

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


TEST(coordinates, toEuclidean_axis_angle_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}), g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}), g(pi/3), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}), g(pi/3), 1u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Degrees{}), g(60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Degrees{}), g(30.), 1u)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}), g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Axis{}), g(3., 2.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Axis{}), g(3., 2.), 1u)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Axis{}), g(3., 2.), 1u)), 2., 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Degrees{}), g(-60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Degrees{}), g(-30.), 1u)), -0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, angle::PositiveRadians{}), g(pi / 3, pi / 6), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, angle::PositiveRadians{}), g(pi / 3, pi / 6), 1u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, angle::Radians{}), g(pi / 3, pi / 6), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, angle::Radians{}), g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, angle::Radians{}), g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, angle::Radians{}), g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}, Dimensions<2>{}, angle::Radians{}), g(pi / 3, 3., 4., pi / 6), 5u)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3., pi / 3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g<float>(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<float>(Axis{}, angle::Radians{}), g<float>(3., pi / 3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<long double>(Axis{}, angle::Radians{}), g<long double>(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);
}


TEST(coordinates, toEuclidean_distance_inclination_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}), g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}), g(-3.), 0u)), -3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}), g(pi/3), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Radians{}), g(pi/3), 1u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Degrees{}), g(60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(angle::Degrees{}), g(30.), 1u)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}), g(-3.), 0u)), -3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, Distance{}), g(3., -2.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, Distance{}), g(3., -2.), 1u)), -2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, Distance{}), g(3., -2.), 1u)), -2., 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Degrees{}), g(-60.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Degrees{}), g(-30.), 1u)), -0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Radians{}, inclination::Radians{}), g(pi / 3, pi / 6), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Radians{}, inclination::Radians{}), g(pi / 3, pi / 6), 1u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Radians{}, inclination::Radians{}), g(pi / 3, pi / 6), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Radians{}, inclination::Radians{}), g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(inclination::Radians{}, inclination::Radians{}), g(pi / 3, pi / 6), 3u)), 0.5, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g(3., pi / 3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}, Distance{}), g(3., pi / 3, 4), 2u)), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}, Distance{}), g(3., pi / 3, 4), 3u)), 4, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g<double>(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g<double>(3., pi / 3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<float>(Distance{}, inclination::Radians{}), g<float>(3., pi / 3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<long double>(Distance{}, inclination::Radians{}), g<long double>(3., pi / 3), 2u)), std::sqrt(3) / 2, 1e-6);
}


TEST(coordinates, toEuclidean_polar_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<Distance, angle::Radians>{}), g(3., pi/3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<Distance, angle::Radians>{}), g(3., pi/3), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<Distance, angle::Radians>{}), g(3., pi/3), 2u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<angle::Radians, Distance>{}), g(pi/3, 3.), 0u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<angle::Radians, Distance>{}), g(pi/3, 3.), 1u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<angle::Radians, Distance>{}), g(pi/3, 3.), 2u)), 3., 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<Distance, angle::Degrees>{}), g(3., 60.), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<Distance, angle::Degrees>{}), g(3., 60.), 1u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Polar<Distance, angle::Degrees>{}), g(3., 60.), 2u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Degrees>{}), g(1.1, 3., 60.), 0u)), 1.1, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Degrees>{}), g(1.1, 3., 60.), 1u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Degrees>{}), g(1.1, 3., 60.), 2u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Degrees>{}), g(1.1, 3., 60.), 3u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Degrees>{}), g<long double>(1.1, 3., 60.), 0u)), 1.1, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Degrees>{}), g<double>(1.1, 3., 60.), 1u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<float>(Axis{}, Polar<Distance, angle::Degrees>{}), g<float>(1.1, 3., 60.), 2u)), 0.5, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<long double>(Axis{}, Polar<Distance, angle::Degrees>{}), g<long double>(1.1, 3., 60.), 3u)), std::sqrt(3)/2, 1e-6);
}


TEST(coordinates, toEuclidean_spherical_dynamic)
{
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., pi/6, pi/3), 0u)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., pi/6, pi/3), 1u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., pi/6, pi/3), 2u)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., pi/6, pi/3), 3u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., pi/6, pi/3), 0u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., pi/6, pi/3), 1u)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., pi/6, pi/3), 2u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., pi/6, pi/3), 3u)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., pi/6, pi/3), 4u)), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 0u)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 1u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 2u)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 3u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 4u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 4u)), 3., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., pi/6, pi/3, 3.), 4u)), 3., 1e-6);

  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<float>(2., pi/6, pi/3, 3.), 0u)), 2., 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<double>(2., pi/6, pi/3, 3.), 1u)), std::sqrt(3)/4, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<float>(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<float>(2., pi/6, pi/3, 3.), 2u)), 0.25, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector<long double>(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<long double>(2., pi/6, pi/3, 3.), 3u)), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((to_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<long double>(2., pi/6, pi/3, 3.), 4u)), 3., 1e-6);
}


TEST(coordinates, fromEuclidean_axis_angle_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}), g(3.), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}), g(-3.), 0u)), -3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(angle::Radians{}), g(0.5, std::sqrt(3) / 2), 0u)), pi / 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(angle::Degrees{}), g(0.5, std::sqrt(3) / 2), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3, 0.5, -std::sqrt(3) / 2), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g(3, 0.5, -std::sqrt(3) / 2), 1u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(angle::Radians{}, Axis{}), g(0.5, std::sqrt(3) / 2, 3), 0u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(angle::Radians{}, Axis{}), g(0.5, std::sqrt(3) / 2, 3), 1u)), 3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g<float>(3, 0.5, -std::sqrt(3) / 2), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, angle::Radians{}), g<double>(3, 0.5, -std::sqrt(3) / 2), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<float>(Axis{}, angle::Radians{}), g<float>(3, 0.5, -std::sqrt(3) / 2), 1u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<long double>(Axis{}, angle::Radians{}), g<long double>(3, 0.5, -std::sqrt(3) / 2), 1u)), -pi/3, 1e-6);
}


TEST(coordinates, fromEuclidean_distance_inclination_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Distance{}), g(3.), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Distance{}), g(-3.), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(inclination::Radians{}), g(0.5, std::sqrt(3) / 2), 0u)), pi / 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(inclination::Degrees{}), g(0.5, std::sqrt(3) / 2), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g(3, 0.5, -std::sqrt(3) / 2), 0u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Distance{}, inclination::Radians{}), g(3, 0.5, -std::sqrt(3) / 2), 1u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(inclination::Radians{}, Distance{}), g(0.5, std::sqrt(3) / 2, 3), 0u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(inclination::Radians{}, Distance{}), g(0.5, std::sqrt(3) / 2, 3), 1u)), 3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(inclination::Radians{}, Distance{}), g<double>(0.5, std::sqrt(3) / 2, 3), 0u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(inclination::Radians{}, Distance{}), g<double>(0.5, std::sqrt(3) / 2, 3), 0u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<float>(inclination::Radians{}, Distance{}), g<float>(0.5, std::sqrt(3) / 2, 3), 1u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<long double>(inclination::Radians{}, Distance{}), g<long double>(0.5, std::sqrt(3) / 2, 3), 1u)), 3, 1e-6);
}


TEST(coordinates, fromEuclidean_polar_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<Distance, angle::Radians>{}), g(3., 0.5, std::sqrt(3) / 2), 0u)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<Distance, angle::Radians>{}), g(3., 0.5, std::sqrt(3) / 2), 1u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(1.1, 3, 0.5, -std::sqrt(3) / 2), 0u)), 1.1, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(1.1, 3, 0.5, -std::sqrt(3) / 2), 1u)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(1.1, 3, 0.5, -std::sqrt(3) / 2), 2u)), -pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(0.5, std::sqrt(3)/2, 3, 1.1), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(0.5, std::sqrt(3)/2, 3, 1.1), 1u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(0.5, std::sqrt(3)/2, 3, 1.1), 2u)), 1.1, 1e-6);

  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g<long double>(0.5, std::sqrt(3)/2, 3, 1.1), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g<double>(0.5, std::sqrt(3)/2, 3, 1.1), 0u)), 60, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<float>(Polar<angle::Degrees, Distance>{}, Axis{}), g<float>(0.5, std::sqrt(3)/2, 3, 1.1), 1u)), 3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<long double>(Polar<angle::Degrees, Distance>{}, Axis{}), g<long double>(0.5, std::sqrt(3)/2, 3, 1.1), 2u)), 1.1, 1e-6);
}


TEST(coordinates, fromEuclidean_spherical_dynamic)
{
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0u)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1u)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2u)), pi/3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 0u)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 1u)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 2u)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., 2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2), 3u)), pi/3, 1e-6);

  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0u)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1u)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3u)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3u)), 3., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3u)), 3., 1e-6);

  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<float>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 0u)), 2., 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<double>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 1u)), pi/6, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<float>(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<float>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 2u)), pi/3, 1e-6);
  EXPECT_NEAR((from_euclidean_element(make_pattern_vector<long double>(Spherical<Distance, angle::Radians, inclination::Radians>{}, Axis{}), g<long double>(2., std::sqrt(3)/4, 0.25, std::sqrt(3)/2, 3.), 3u)), 3., 1e-6);
}


TEST(coordinates, get_wrapped_component_axis_angle_dynamic)
{
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}), g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}), g(-3.), 0u)), -3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}), g(1.2*pi), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::PositiveRadians{}), g(-0.2*pi), 0u)), 1.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Degrees{}), g(200.), 0u)), -160., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::PositiveDegrees{}), g(-20.), 0u)), 340., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Circle{}), g(-0.2), 0u)), 0.8, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, angle::Radians{}), g(3, 1.2*pi), 0u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, angle::Radians{}), g(3, 1.2*pi), 1u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}), g(1.2*pi, 3), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}), g(1.2*pi, 3), 1u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g(1.2*pi, 5, 1.3*pi, 3), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g(1.2*pi, 5, 1.3*pi, 3), 1u)), 5, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g(1.2*pi, 5, 1.3*pi, 3), 2u)), -0.7*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g(1.2*pi, 5, 1.3*pi, 3), 3u)), 3, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g<double>(1.2*pi, 5, 1.3*pi, 3), 0u)), -0.8*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g<double>(1.2*pi, 5, 1.3*pi, 3), 1u)), 5, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<float>(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g<float>(1.2*pi, 5, 1.3*pi, 3), 2u)), -0.7*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<long double>(angle::Radians{}, Axis{}, angle::Radians{}, Axis{}), g<long double>(1.2*pi, 5, 1.3*pi, 3), 3u)), 3, 1e-6);
}


TEST(coordinates, get_wrapped_component_distance_inclination_dynamic)
{
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Distance{}), g(3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Distance{}), g(-3.), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(inclination::Radians{}), g(0.7*pi), 0u)), 0.3*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(inclination::Radians{}), g(1.2*pi), 0u)), -0.2*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(inclination::Degrees{}), g(110.), 0u)), 70., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(inclination::Degrees{}), g(200.), 0u)), -20., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Distance{}, inclination::Radians{}), g(3, 0.7*pi), 0u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Distance{}, inclination::Radians{}), g(3, 0.7*pi), 1u)), 0.3*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(inclination::Radians{}, Distance{}), g(1.2*pi, 3), 0u)), -0.2*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(inclination::Radians{}, Distance{}), g(1.2*pi, 3), 1u)), 3, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Distance{}, inclination::Radians{}), g<long double>(3, 0.7*pi), 0u)), 3, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Distance{}, inclination::Radians{}), g<double>(3, 0.7*pi), 1u)), 0.3*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<float>(inclination::Radians{}, Distance{}), g<float>(1.2*pi, 3), 0u)), -0.2*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<long double>(inclination::Radians{}, Distance{}), g<long double>(1.2*pi, 3), 1u)), 3, 1e-6);
}


TEST(coordinates, get_wrapped_component_polar_dynamic)
{
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(3., -2., 1.1*pi), 0u)), 3., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(3., -2., 1.1*pi), 1u)), 2., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(3., 2., 1.1*pi), 2u)), -0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<Distance, angle::Radians>{}), g(3., -2., 1.1*pi), 2u)), 0.1*pi, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(190., 2., 4.), 0u)), -170, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(190., -2., 4.), 0u)), 10, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(190., -2., 4.), 1u)), 2, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<angle::Degrees, Distance>{}, Axis{}), g(190., -2., -4.), 2u)), -4, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Axis{}), g(-2., -0.1*pi, 4.), 0u)), 2, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Axis{}), g(2., -0.1*pi, 4.), 1u)), 1.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Axis{}), g(-2., -0.1*pi, 4.), 1u)), 0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Axis{}), g(-2., -0.1*pi, -4.), 2u)), -4, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g(5., -10., 2.), 0u)), 5, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g(5., -10., 2.), 1u)), 350, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g(5., -10., -2.), 1u)), 170, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g(5., -10., -2.), 2u)), 2, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g<float>(5., -10., 2.), 0u)), 5, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g<double>(5., -10., 2.), 1u)), 350, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<float>(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g<float>(5., -10., -2.), 1u)), 170, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<long double>(Axis{}, Polar<angle::PositiveDegrees, Distance>{}), g<long double>(5., -10., -2.), 2u)), 2, 1e-6);
}


TEST(coordinates, get_wrapped_component_spherical_dynamic)
{
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., 1.1*pi, 0.6*pi), 0u)), 2., 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., 0.5*pi, 0.6*pi), 1u)), -0.5*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., 0.5*pi, 0.6*pi), 2u)), 0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(-2., 0.5*pi, 0.6*pi), 1u)), 0.5*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(-2., 0.5*pi, 0.6*pi), 2u)), -0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., 1.1*pi, 0.6*pi), 1u)), 0.1*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(2., 1.1*pi, 0.6*pi), 2u)), 0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(-2., 1.1*pi, 0.6*pi), 1u)), -0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g(-2., 1.1*pi, 0.6*pi), 2u)), -0.4*pi, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., -2., 1.1*pi, 0.6*pi), 3u)), -0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), g(3., -2., 1.1*pi, 0.6*pi), 3u)), -0.4*pi, 1e-6);

  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g<float>(2., 1.1*pi, 0.6*pi), 1u)), 0.1*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}), g<double>(2., 1.1*pi, 0.6*pi), 2u)), 0.4*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<float>(Spherical<Distance, angle::Radians, inclination::Radians>{}), g<float>(-2., 1.1*pi, 0.6*pi), 1u)), -0.9*pi, 1e-6);
  EXPECT_NEAR((get_wrapped_component(make_pattern_vector<long double>(Spherical<Distance, angle::Radians, inclination::Radians>{}), g<long double>(-2., 1.1*pi, 0.6*pi), 2u)), -0.4*pi, 1e-6);
}


TEST(coordinates, set_wrapped_component_axis_angle_dynamic)
{
  std::array<double, 2> a1 = {0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) { return a1[i]; }};

  set_wrapped_component(make_pattern_vector(Axis{}, Axis{}), s, g, 2.1, 1u);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, angle::Radians{}), s, g, 1.2*pi, 1u);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(angle::PositiveRadians{}, Axis{}), s, g, -0.2*pi, 0u);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(angle::Degrees{}, Axis{}), s, g, 200., 0u);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, angle::PositiveDegrees{}), s, g, -20., 1u);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, angle::Circle{}), s, g, -0.2, 1u);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);

  set_wrapped_component(make_pattern_vector(Axis{}, Axis{}), s, g, 2.1, 1u);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, angle::Radians{}), s, g, 1.2*pi, 1u);
  EXPECT_NEAR(a1[1], -0.8*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(angle::PositiveRadians{}, Axis{}), s, g, -0.2*pi, 0u);
  EXPECT_NEAR(a1[0], 1.8*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(angle::Degrees{}, Axis{}), s, g, 200., 0u);
  EXPECT_NEAR(a1[0], -160, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, angle::PositiveDegrees{}), s, g, -20., 1u);
  EXPECT_NEAR(a1[1], 340, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, angle::Circle{}), s, g, -0.2, 1u);
  EXPECT_NEAR(a1[1], 0.8, 1e-6);
}


TEST(coordinates, set_wrapped_component_distance_inclination_dynamic)
{
  std::array<float, 2> a1 = {0, 0};
  auto s = std::function {[&](const float& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  set_wrapped_component(make_pattern_vector(Distance{}, Distance{}), s, g, 2.1, 0u);
  set_wrapped_component(make_pattern_vector(Distance{}, Distance{}), s, g, -2.1, 1u);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  set_wrapped_component(make_pattern_vector(inclination::Radians{}, inclination::Degrees{}), s, g, 0.7*pi, 0u);
  set_wrapped_component(make_pattern_vector(inclination::Radians{}, inclination::Degrees{}), s, g, 110., 1u);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  set_wrapped_component(make_pattern_vector(inclination::Degrees{}, inclination::Radians{}), s, g, 200., 0u);
  set_wrapped_component(make_pattern_vector(inclination::Degrees{}, inclination::Radians{}), s, g, 1.2*pi, 1u);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);

  set_wrapped_component(make_pattern_vector<float>(Distance{}, Distance{}), s, g, 2.1, 0u);
  set_wrapped_component(make_pattern_vector<float>(Distance{}, Distance{}), s, g, -2.1, 1u);
  EXPECT_NEAR(a1[0], 2.1, 1e-6);
  EXPECT_NEAR(a1[1], 2.1, 1e-6);
  set_wrapped_component(make_pattern_vector<float>(inclination::Radians{}, inclination::Degrees{}), s, g, 0.7*pi, 0u);
  set_wrapped_component(make_pattern_vector<float>(inclination::Radians{}, inclination::Degrees{}), s, g, 110., 1u);
  EXPECT_NEAR(a1[0], 0.3*pi, 1e-6);
  EXPECT_NEAR(a1[1], 70, 1e-6);
  set_wrapped_component(make_pattern_vector<float>(inclination::Degrees{}, inclination::Radians{}), s, g, 200., 0u);
  set_wrapped_component(make_pattern_vector<float>(inclination::Degrees{}, inclination::Radians{}), s, g, 1.2*pi, 1u);
  EXPECT_NEAR(a1[0], -20., 1e-6);
  EXPECT_NEAR(a1[1], -0.2*pi, 1e-6);
}


TEST(coordinates, set_wrapped_component_polar_dynamic)
{
  std::array<long double, 4> a1 = {1, -0.2*numbers::pi_v<long double>, 0, 1};
  auto s = std::function {[&](const long double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::Radians>{}, Polar<angle::Degrees, Distance>{}), s, g, -2., 0u);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  EXPECT_NEAR(a1[1], 0.8*numbers::pi_v<long double>, 1e-6);
  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::Radians>{}, Polar<angle::Degrees, Distance>{}), s, g, 1.1*numbers::pi_v<long double>, 1u);
  EXPECT_NEAR(a1[1], -0.9*numbers::pi_v<long double>, 1e-6);
  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::Radians>{}, Polar<angle::Degrees, Distance>{}), s, g, 190., 2u);
  EXPECT_NEAR(a1[2], -170, 1e-6);
  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::Radians>{}, Polar<angle::Degrees, Distance>{}), s, g, -2., 3u);
  EXPECT_NEAR(a1[2], 10, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -2., 0u);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -0.1*numbers::pi_v<long double>, 1u);
  EXPECT_NEAR(a1[1], 1.9*numbers::pi_v<long double>, 1e-6);
  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -10., 2u);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  set_wrapped_component(make_pattern_vector(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -2., 3u);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(make_pattern_vector<long double>(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -2., 0u);
  EXPECT_NEAR(a1[0], 2, 1e-6);
  set_wrapped_component(make_pattern_vector<long double>(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -0.1*numbers::pi_v<long double>, 1u);
  EXPECT_NEAR(a1[1], 1.9*numbers::pi_v<long double>, 1e-6);
  set_wrapped_component(make_pattern_vector<long double>(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -10., 2u);
  EXPECT_NEAR(a1[2], 350, 1e-6);
  set_wrapped_component(make_pattern_vector<long double>(Polar<Distance, angle::PositiveRadians>{}, Polar<angle::PositiveDegrees, Distance>{}), s, g, -2., 3u);
  EXPECT_NEAR(a1[2], 170, 1e-6);
  EXPECT_NEAR(a1[3], 2, 1e-6);
}


TEST(coordinates, set_wrapped_component_spherical_dynamic)
{
  std::array<double, 4> a1 = {0, 0, 0, 0};
  auto s = std::function {[&](const double& s, std::size_t i) { a1[i] = s; }};
  auto g = std::function {[&](std::size_t i) {return a1[i]; }};

  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, -2., 1u);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, 1.1*pi, 2u);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, 0.6*pi, 3u);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.4*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, -3., 1u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, -0.6*pi, 3u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], 0.1*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, angle::Radians, inclination::Radians>{}), s, g, -2.6*pi, 3u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.9*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.4*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, inclination::Radians, angle::Radians>{}), s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, inclination::Radians, angle::Radians>{}), s, g, -2., 1u);
  EXPECT_NEAR(a1[1], 2, 1e-6);
  EXPECT_NEAR(a1[3], -pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, inclination::Radians, angle::Radians>{}), s, g, 0.6*pi, 2u);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.0, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, inclination::Radians, angle::Radians>{}), s, g, 1.1*pi, 3u);
  EXPECT_NEAR(a1[2], 0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, inclination::Radians, angle::Radians>{}), s, g, -3., 1u);
  EXPECT_NEAR(a1[1], 3, 1e-6);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], 0.1*pi, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<Distance, inclination::Radians, angle::Radians>{}), s, g, 3.4*pi, 2u);
  EXPECT_NEAR(a1[2], -0.4*pi, 1e-6);
  EXPECT_NEAR(a1[3], -0.9*pi, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -2., 2u);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -170., 1u);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, 100., 3u);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, 550., 1u);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -3., 2u);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);

  a1 = {0, 0, 0, 0};

  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -3.1, 0u);
  EXPECT_NEAR(a1[0], -3.1, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -2., 2u);
  EXPECT_NEAR(a1[2], 2, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -170., 1u);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, 100., 3u);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, 550., 1u);
  EXPECT_NEAR(a1[1], 190, 1e-6);
  EXPECT_NEAR(a1[3], 80, 1e-6);
  set_wrapped_component(make_pattern_vector(Axis{}, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>{}), s, g, -3., 2u);
  EXPECT_NEAR(a1[1], 10, 1e-6);
  EXPECT_NEAR(a1[2], 3, 1e-6);
  EXPECT_NEAR(a1[3], -80, 1e-6);
}

