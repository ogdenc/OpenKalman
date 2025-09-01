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
 * \brief Tests for \ref coordinates::compare_with
 */

#include "collections/tests/tests.hpp"
#include "coordinates/descriptors/Dimensions.hpp"
#include "coordinates/descriptors/Distance.hpp"
#include "coordinates/descriptors/Angle.hpp"
#include "coordinates/descriptors/Inclination.hpp"
#include "coordinates/descriptors/Polar.hpp"
#include "coordinates/descriptors/Spherical.hpp"
#include "coordinates/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

#include "coordinates/concepts/compares_with.hpp"

TEST(coordinates, compares_with_euclidean_descriptors)
{
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>&, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdcompat::is_lt>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdcompat::is_gt>);
  static_assert(compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdcompat::is_eq, applicability::permitted>);

  static_assert(compares_with<Axis, Dimensions<1>>);
  static_assert(compares_with<Dimensions<8>, Dimensions<8>>);
  static_assert(compares_with<Dimensions<8>, Dimensions<9>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<8>, Dimensions<9>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<8>, Dimensions<9>, &stdcompat::is_neq>);
  static_assert(compares_with<Dimensions<9>, Dimensions<8>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<9>, Dimensions<8>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<9>, Dimensions<8>, &stdcompat::is_neq>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<8>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<9>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<9>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<9>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<9>, Dimensions<8>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<9>, Dimensions<8>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<9>, Dimensions<8>, &stdcompat::is_eq, applicability::permitted>);

  static_assert(compares_with<Dimensions<2>, unsigned, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdcompat::is_neq>);

  static_assert(compares_with<unsigned, Dimensions<2>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdcompat::is_eq>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdcompat::is_lt>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdcompat::is_lteq>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdcompat::is_gt>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdcompat::is_gteq>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdcompat::is_neq>);

  static_assert(compares_with<std::size_t, std::size_t, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdcompat::is_eq>);
  static_assert(compares_with<std::size_t, std::size_t, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdcompat::is_lt>);
  static_assert(compares_with<std::size_t, std::size_t, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdcompat::is_lteq>);
  static_assert(compares_with<std::size_t, std::size_t, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdcompat::is_gt>);
  static_assert(compares_with<std::size_t, std::size_t, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdcompat::is_gteq>);
  static_assert(compares_with<std::size_t, std::size_t, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdcompat::is_neq>);

  static_assert(compares_with<std::integral_constant<std::size_t, 0>, std::integral_constant<std::size_t, 0>, &stdcompat::is_eq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 0>, std::integral_constant<std::size_t, 0>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 0>, std::integral_constant<std::size_t, 0>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<0>, std::integral_constant<std::size_t, 0>, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<0>, std::integral_constant<std::size_t, 0>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<0>, std::integral_constant<std::size_t, 0>, &stdcompat::is_gteq>);

  static_assert(compares_with<Dimensions<0>, std::size_t, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdcompat::is_lteq>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdcompat::is_neq>);

  static_assert(compares_with<std::size_t, Dimensions<0>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdcompat::is_eq>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdcompat::is_gt>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdcompat::is_neq>);

  static_assert(collections::size_of_v<stdcompat::ranges::empty_view<Dimensions<100>>> == 0);
  static_assert(compares_with<stdcompat::ranges::empty_view<Dimensions<1>>, Dimensions<0>>);
  static_assert(compares_with<stdcompat::ranges::empty_view<Dimensions<0>>, Dimensions<0>>);
  static_assert(compares_with<stdcompat::ranges::empty_view<Polar<>>, Dimensions<0>>);
}


TEST(coordinates, compares_with_misc_descriptors)
{
  static_assert(compares_with<Distance, Distance>);
  static_assert(compares_with<Angle<>, Angle<>>);
  static_assert(compares_with<angle::Degrees, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>>);
  static_assert(compares_with<Inclination<>, Inclination<>>);
  static_assert(compares_with<inclination::Degrees, Inclination<std::integral_constant<int, 180>>>);
  static_assert(compares_with<Polar<>, Polar<>>);
  static_assert(compares_with<Polar<Distance, angle::Degrees>, Polar<Distance, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>>>);
  static_assert(compares_with<Spherical<>, Spherical<>>);
  static_assert(compares_with<Spherical<Distance, inclination::Degrees, angle::Degrees>, Spherical<Distance, Inclination<std::integral_constant<int, 180>>, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>>>);

  static_assert(compares_with<stdcompat::reference_wrapper<Distance>, stdcompat::reference_wrapper<Distance>>);
  static_assert(compares_with<stdcompat::reference_wrapper<Polar<>>, Polar<>>);
  static_assert(compares_with<Spherical<>, stdcompat::reference_wrapper<Spherical<>>>);

  static_assert(not compares_with<Axis, angle::Radians, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Axis, angle::Radians, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<Axis, angle::Radians, &stdcompat::is_neq>);
  static_assert(not compares_with<angle::Degrees, angle::Radians, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, angle::Radians, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<angle::Degrees, angle::Radians, &stdcompat::is_neq>);
  static_assert(not compares_with<Axis, Polar<>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Axis, Polar<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<Axis, Polar<>, &stdcompat::is_neq>);
  static_assert(not compares_with<Polar<>, angle::Radians, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Polar<>, angle::Radians, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<Polar<>, angle::Radians, &stdcompat::is_neq>);

  static_assert(not compares_with<angle::Degrees, unsigned, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(compares_with<angle::Degrees, unsigned, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdcompat::is_gteq>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<angle::Degrees, unsigned, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdcompat::is_gt>);
  static_assert(compares_with<angle::Degrees, unsigned, &stdcompat::is_neq>);

  static_assert(not compares_with<unsigned, angle::Radians, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<unsigned, angle::Radians, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdcompat::is_lteq>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<unsigned, angle::Radians, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdcompat::is_lt>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<unsigned, angle::Radians, &stdcompat::is_neq>);

  static_assert(compares_with<Any<double>, Any<double>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdcompat::is_eq>);
  static_assert(compares_with<Any<double>, Any<double>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdcompat::is_lt>);
  static_assert(compares_with<Any<double>, Any<double>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdcompat::is_gt>);
  static_assert(compares_with<Any<double>, Any<double>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdcompat::is_lteq>);
  static_assert(compares_with<Any<double>, Any<double>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdcompat::is_gteq>);
  static_assert(compares_with<Any<double>, Any<double>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdcompat::is_neq>);

  static_assert(compares_with<Any<double>, Any<float>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdcompat::is_eq>);
  static_assert(compares_with<Any<double>, Any<float>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdcompat::is_lt>);
  static_assert(compares_with<Any<double>, Any<float>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdcompat::is_gt>);
  static_assert(compares_with<Any<double>, Any<float>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdcompat::is_lteq>);
  static_assert(compares_with<Any<double>, Any<float>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdcompat::is_gteq>);
  static_assert(compares_with<Any<double>, Any<float>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdcompat::is_neq>);

  static_assert(compares_with<Distance, Any<double>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdcompat::is_eq>);
  static_assert(not compares_with<Distance, Any<double>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<Distance, Any<double>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdcompat::is_gt>);
  static_assert(compares_with<Distance, Any<double>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdcompat::is_lteq>);
  static_assert(compares_with<Distance, Any<double>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdcompat::is_gteq>);
  static_assert(compares_with<Distance, Any<double>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdcompat::is_neq>);

  static_assert(compares_with<Any<double>, Distance, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdcompat::is_eq>);
  static_assert(compares_with<Any<double>, Distance, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdcompat::is_lt>);
  static_assert(not compares_with<Any<double>, Distance, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<Any<double>, Distance, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdcompat::is_lteq>);
  static_assert(compares_with<Any<double>, Distance, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdcompat::is_gteq>);
  static_assert(compares_with<Any<double>, Distance, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdcompat::is_neq>);

  static_assert(compares_with<Dimensions<2>, Any<double>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdcompat::is_neq>);

  static_assert(compares_with<Any<double>, Dimensions<2>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdcompat::is_eq>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdcompat::is_lt>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdcompat::is_gt>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdcompat::is_lteq>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdcompat::is_gteq>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdcompat::is_neq>);

  static_assert(compares_with<Any<double>, unsigned, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdcompat::is_eq>);
  static_assert(compares_with<Any<double>, unsigned, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdcompat::is_lt>);
  static_assert(compares_with<Any<double>, unsigned, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdcompat::is_gt>);
  static_assert(compares_with<Any<double>, unsigned, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdcompat::is_lteq>);
  static_assert(compares_with<Any<double>, unsigned, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdcompat::is_gteq>);
  static_assert(compares_with<Any<double>, unsigned, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdcompat::is_neq>);

  static_assert(compares_with<unsigned, Any<double>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdcompat::is_eq>);
  static_assert(compares_with<unsigned, Any<double>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdcompat::is_lt>);
  static_assert(compares_with<unsigned, Any<double>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdcompat::is_gt>);
  static_assert(compares_with<unsigned, Any<double>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdcompat::is_lteq>);
  static_assert(compares_with<unsigned, Any<double>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdcompat::is_gteq>);
  static_assert(compares_with<unsigned, Any<double>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdcompat::is_neq>);
}


TEST(coordinates, compares_with_fixed_size_collection_fixed)
{
  static_assert(compares_with<std::tuple<>, std::tuple<>>);
  static_assert(compares_with<std::tuple<>, std::tuple<>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<>, std::tuple<>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<>, std::tuple<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::tuple<>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::tuple<>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<std::tuple<>, Dimensions<0>>);
  static_assert(compares_with<std::tuple<>, Dimensions<0>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<>, Dimensions<0>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<>, Dimensions<0>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, Dimensions<0>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, Dimensions<0>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>, &stdcompat::is_gteq>);
  static_assert(not compares_with<Dimensions<0>, std::tuple<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::tuple<>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::tuple<>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdcompat::is_lteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdcompat::is_lteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdcompat::is_lt>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_neq>);

  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdcompat::is_lteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdcompat::is_lt>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdcompat::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdcompat::is_neq>);
}


TEST(coordinates, compares_with_fixed_size_collection_dynamic)
{
  static_assert(compares_with<std::tuple<>, std::size_t, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdcompat::is_lteq>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdcompat::is_neq>);

  static_assert(compares_with<std::size_t, std::tuple<>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdcompat::is_eq>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdcompat::is_gt>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lt>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdcompat::is_neq>);
}


TEST(coordinates, compares_with_dynamic_size_collection)
{
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_lteq>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdcompat::is_neq>);

  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdcompat::is_neq>);

  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdcompat::is_neq>);

  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdcompat::is_neq>);

  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_eq>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdcompat::is_neq>);

  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_lteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_gteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_lt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_gt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdcompat::is_neq>);

  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdcompat::is_neq>);

  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_gt>);
  static_assert(compares_with<Distance, std::vector<Dimensions<1>>, &stdcompat::is_neq>);

  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_lteq>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Distance, &stdcompat::is_neq>);

  static_assert(compares_with<Distance, std::vector<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Any<>>, Distance, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdcompat::is_neq>);

  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdcompat::is_neq>);

  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_lteq>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_lt>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdcompat::is_neq>);
}


TEST(coordinates, compares_with_unsized_collection)
{
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_eq>);

  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Dimensions<0>>, &stdcompat::is_neq>);

  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lt>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<0>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<2>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<2>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<2>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<2>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<2>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<2>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<3>>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<>>, &stdcompat::is_gteq>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Dimensions<>>, stdcompat::ranges::repeat_view<Dimensions<>>, &stdcompat::is_neq, applicability::permitted>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_lt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, stdcompat::ranges::repeat_view<Distance>, &stdcompat::is_neq>);

  // In theory, a boundless collection of Any<> could be Distance in the first element and Dimensions<0> otherwise.
  static_assert(compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_lt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, Distance, &stdcompat::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_lt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdcompat::is_neq>);

  static_assert(compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_eq>);
  static_assert(compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gteq>);
  static_assert(compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_lt>);
  static_assert(compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_gt>);
  static_assert(compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdcompat::ranges::repeat_view<Any<>>, &stdcompat::is_neq>);

  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_eq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_lteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_gteq>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_lt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_gt>);
  static_assert(compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdcompat::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdcompat::is_neq>);
}
