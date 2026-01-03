/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref patterns::compare_with
 */

#include "collections/tests/tests.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Distance.hpp"
#include "patterns/descriptors/Angle.hpp"
#include "patterns/descriptors/Inclination.hpp"
#include "patterns/descriptors/Polar.hpp"
#include "patterns/descriptors/Spherical.hpp"
#include "patterns/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;

#include "patterns/concepts/compares_with.hpp"

TEST(patterns, compares_with_euclidean_descriptors)
{
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>&, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>, &stdex::is_lteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>, &stdex::is_gteq>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>, &stdex::is_neq, applicability::permitted>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdex::is_lt>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdex::is_lteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdex::is_neq>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdex::is_gt>);
  static_assert(compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdex::is_gteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdex::is_neq>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 2>, &stdex::is_eq, applicability::permitted>);

  static_assert(compares_with<Axis, Dimensions<1>>);
  static_assert(compares_with<Dimensions<8>, Dimensions<8>>);
  static_assert(compares_with<Dimensions<8>, Dimensions<9>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<8>, Dimensions<9>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<8>, Dimensions<9>, &stdex::is_neq>);
  static_assert(compares_with<Dimensions<9>, Dimensions<8>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<9>, Dimensions<8>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<9>, Dimensions<8>, &stdex::is_neq>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<8>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<9>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<9>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<8>, Dimensions<9>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<9>, Dimensions<8>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<9>, Dimensions<8>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<9>, Dimensions<8>, &stdex::is_eq, applicability::permitted>);

  static_assert(compares_with<Dimensions<2>, unsigned, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<2>, unsigned, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, unsigned, &stdex::is_neq>);

  static_assert(compares_with<unsigned, Dimensions<2>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdex::is_eq>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdex::is_lt>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdex::is_lteq>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdex::is_gt>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdex::is_gteq>);
  static_assert(compares_with<unsigned, Dimensions<2>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Dimensions<2>, &stdex::is_neq>);

  static_assert(compares_with<std::size_t, std::size_t, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdex::is_eq>);
  static_assert(compares_with<std::size_t, std::size_t, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdex::is_lt>);
  static_assert(compares_with<std::size_t, std::size_t, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdex::is_lteq>);
  static_assert(compares_with<std::size_t, std::size_t, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdex::is_gt>);
  static_assert(compares_with<std::size_t, std::size_t, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdex::is_gteq>);
  static_assert(compares_with<std::size_t, std::size_t, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::size_t, &stdex::is_neq>);

  static_assert(compares_with<std::integral_constant<std::size_t, 0>, std::integral_constant<std::size_t, 0>, &stdex::is_eq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 0>, std::integral_constant<std::size_t, 0>, &stdex::is_lteq>);
  static_assert(compares_with<std::integral_constant<std::size_t, 0>, std::integral_constant<std::size_t, 0>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<0>, std::integral_constant<std::size_t, 0>, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<0>, std::integral_constant<std::size_t, 0>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<0>, std::integral_constant<std::size_t, 0>, &stdex::is_gteq>);

  static_assert(compares_with<Dimensions<0>, std::size_t, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdex::is_lteq>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<0>, std::size_t, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::size_t, &stdex::is_neq>);

  static_assert(compares_with<std::size_t, Dimensions<0>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdex::is_eq>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdex::is_gt>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdex::is_gteq>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdex::is_lteq>);
  static_assert(compares_with<std::size_t, Dimensions<0>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, Dimensions<0>, &stdex::is_neq>);

  static_assert(collections::size_of_v<stdex::ranges::empty_view<Dimensions<100>>> == 0);
  static_assert(compares_with<stdex::ranges::empty_view<Dimensions<1>>, Dimensions<0>>);
  static_assert(compares_with<stdex::ranges::empty_view<Dimensions<0>>, Dimensions<0>>);
  static_assert(compares_with<stdex::ranges::empty_view<Polar<>>, Dimensions<0>>);
}


TEST(patterns, compares_with_misc_descriptors)
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

  static_assert(compares_with<stdex::reference_wrapper<Distance>, stdex::reference_wrapper<Distance>>);
  static_assert(compares_with<stdex::reference_wrapper<Polar<>>, Polar<>>);
  static_assert(compares_with<Spherical<>, stdex::reference_wrapper<Spherical<>>>);

  static_assert(not compares_with<Axis, angle::Radians, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Axis, angle::Radians, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<Axis, angle::Radians, &stdex::is_neq>);
  static_assert(not compares_with<angle::Degrees, angle::Radians, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, angle::Radians, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<angle::Degrees, angle::Radians, &stdex::is_neq>);
  static_assert(not compares_with<Axis, Polar<>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Axis, Polar<>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<Axis, Polar<>, &stdex::is_neq>);
  static_assert(not compares_with<Polar<>, angle::Radians, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Polar<>, angle::Radians, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<Polar<>, angle::Radians, &stdex::is_neq>);

  static_assert(not compares_with<angle::Degrees, unsigned, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdex::is_lteq, applicability::permitted>);
  static_assert(compares_with<angle::Degrees, unsigned, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdex::is_gteq>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<angle::Degrees, unsigned, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, &stdex::is_gt>);
  static_assert(compares_with<angle::Degrees, unsigned, &stdex::is_neq>);

  static_assert(not compares_with<unsigned, angle::Radians, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<unsigned, angle::Radians, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdex::is_lteq>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<unsigned, angle::Radians, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdex::is_lt>);
  static_assert(not compares_with<unsigned, angle::Radians, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<unsigned, angle::Radians, &stdex::is_neq>);

  static_assert(compares_with<Any<double>, Any<double>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdex::is_eq>);
  static_assert(compares_with<Any<double>, Any<double>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdex::is_lt>);
  static_assert(compares_with<Any<double>, Any<double>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdex::is_gt>);
  static_assert(compares_with<Any<double>, Any<double>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdex::is_lteq>);
  static_assert(compares_with<Any<double>, Any<double>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdex::is_gteq>);
  static_assert(compares_with<Any<double>, Any<double>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<double>, &stdex::is_neq>);

  static_assert(compares_with<Any<double>, Any<float>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdex::is_eq>);
  static_assert(compares_with<Any<double>, Any<float>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdex::is_lt>);
  static_assert(compares_with<Any<double>, Any<float>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdex::is_gt>);
  static_assert(compares_with<Any<double>, Any<float>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdex::is_lteq>);
  static_assert(compares_with<Any<double>, Any<float>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdex::is_gteq>);
  static_assert(compares_with<Any<double>, Any<float>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Any<float>, &stdex::is_neq>);

  static_assert(compares_with<Distance, Any<double>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdex::is_eq>);
  static_assert(not compares_with<Distance, Any<double>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<Distance, Any<double>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdex::is_gt>);
  static_assert(compares_with<Distance, Any<double>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdex::is_lteq>);
  static_assert(compares_with<Distance, Any<double>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdex::is_gteq>);
  static_assert(compares_with<Distance, Any<double>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Distance, Any<double>, &stdex::is_neq>);

  static_assert(compares_with<Any<double>, Distance, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdex::is_eq>);
  static_assert(compares_with<Any<double>, Distance, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdex::is_lt>);
  static_assert(not compares_with<Any<double>, Distance, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<Any<double>, Distance, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdex::is_lteq>);
  static_assert(compares_with<Any<double>, Distance, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdex::is_gteq>);
  static_assert(compares_with<Any<double>, Distance, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Distance, &stdex::is_neq>);

  static_assert(compares_with<Dimensions<2>, Any<double>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<2>, Any<double>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, Any<double>, &stdex::is_neq>);

  static_assert(compares_with<Any<double>, Dimensions<2>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdex::is_eq>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdex::is_lt>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdex::is_gt>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdex::is_lteq>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdex::is_gteq>);
  static_assert(compares_with<Any<double>, Dimensions<2>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, Dimensions<2>, &stdex::is_neq>);

  static_assert(compares_with<Any<double>, unsigned, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdex::is_eq>);
  static_assert(compares_with<Any<double>, unsigned, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdex::is_lt>);
  static_assert(compares_with<Any<double>, unsigned, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdex::is_gt>);
  static_assert(compares_with<Any<double>, unsigned, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdex::is_lteq>);
  static_assert(compares_with<Any<double>, unsigned, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdex::is_gteq>);
  static_assert(compares_with<Any<double>, unsigned, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Any<double>, unsigned, &stdex::is_neq>);

  static_assert(compares_with<unsigned, Any<double>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdex::is_eq>);
  static_assert(compares_with<unsigned, Any<double>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdex::is_lt>);
  static_assert(compares_with<unsigned, Any<double>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdex::is_gt>);
  static_assert(compares_with<unsigned, Any<double>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdex::is_lteq>);
  static_assert(compares_with<unsigned, Any<double>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdex::is_gteq>);
  static_assert(compares_with<unsigned, Any<double>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<unsigned, Any<double>, &stdex::is_neq>);
}


TEST(patterns, compares_with_fixed_size_collection_fixed)
{
  static_assert(compares_with<std::tuple<>, std::tuple<>>);
  static_assert(compares_with<std::tuple<>, std::tuple<>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<>, std::tuple<>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<>, std::tuple<>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::tuple<>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::tuple<>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<std::tuple<>, Dimensions<0>>);
  static_assert(compares_with<std::tuple<>, Dimensions<0>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<>, Dimensions<0>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<>, Dimensions<0>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, Dimensions<0>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, Dimensions<0>, &stdex::is_neq, applicability::permitted>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>, &stdex::is_gteq>);
  static_assert(not compares_with<Dimensions<0>, std::tuple<>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::tuple<>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<0>, std::tuple<>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<>, &stdex::is_lteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Spherical<>, angle::Radians>, std::tuple<Distance, Polar<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Polar<>, angle::Radians>, std::tuple<Distance, Spherical<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Axis, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Axis, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, Dimensions<0>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<0>, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, angle::Radians>, std::tuple<Distance, Dimensions<5>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);

  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_neq, applicability::permitted>);

  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdex::is_lteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdex::is_lt>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_neq>);

  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdex::is_lteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdex::is_lt>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, &stdex::is_neq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<2>, Dimensions<5>, angle::Radians, Distance, Dimensions<2>>, std::tuple<Distance, Dimensions<3>, Dimensions<4>, angle::Radians>, &stdex::is_neq>);
}


TEST(patterns, compares_with_fixed_size_collection_dynamic)
{
  static_assert(compares_with<std::tuple<>, std::size_t, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdex::is_lteq>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<>, std::size_t, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<>, std::size_t, &stdex::is_neq>);

  static_assert(compares_with<std::size_t, std::tuple<>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdex::is_eq>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdex::is_gt>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdex::is_gteq>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdex::is_lteq>);
  static_assert(compares_with<std::size_t, std::tuple<>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::size_t, std::tuple<>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, angle::Radians>, std::tuple<Distance, unsigned, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lt>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, angle::Radians, unsigned>, std::tuple<Distance, angle::Radians, unsigned>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Any<>, angle::Radians>, std::tuple<Distance, unsigned, Any<>, angle::Radians>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, unsigned, Any<>, angle::Radians>, std::tuple<Distance, Any<>, angle::Radians>, &stdex::is_neq>);
}


TEST(patterns, compares_with_dynamic_size_collection)
{
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdex::is_gteq>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Dimensions<0>>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_lteq>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<0>>, std::vector<Distance>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_gteq>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::vector<Dimensions<0>>, &stdex::is_neq>);

  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Distance>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, Dimensions<>, &stdex::is_neq>);

  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<>, std::vector<Dimensions<1>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Dimensions<>, &stdex::is_neq>);

  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<2>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, Dimensions<2>, &stdex::is_neq>);

  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_eq>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<6>, std::vector<Dimensions<3>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<6>, &stdex::is_neq>);

  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_lteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_gteq>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_lt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_gt>);
  static_assert(compares_with<Dimensions<2>, std::vector<Dimensions<3>>, &stdex::is_neq>);

  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<3>>, Dimensions<2>, &stdex::is_neq>);

  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_gteq>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_gt>);
  static_assert(compares_with<Distance, std::vector<Dimensions<1>>, &stdex::is_neq>);

  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_lteq>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, Distance, &stdex::is_neq>);

  static_assert(compares_with<Distance, std::vector<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<Distance, std::vector<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Distance, std::vector<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Any<>>, Distance, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Any<>>, Distance, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, Distance, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Any<>>, std::vector<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<1>>, std::vector<Dimensions<1>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<6>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<6>>, std::vector<Dimensions<2>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<2>>, std::vector<Dimensions<5>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Dimensions<5>>, std::vector<Dimensions<2>>, &stdex::is_neq>);

  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_lteq, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_gteq>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Dimensions<1>>, std::vector<Distance>, &stdex::is_neq>);

  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_lteq>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_lt>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Dimensions<1>>, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Distance>, std::vector<Distance>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, std::tuple<Distance, Distance>, &stdex::is_neq>);
}


TEST(patterns, compares_with_unsized_collection)
{
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_eq>);

  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_gteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Dimensions<0>>, &stdex::is_neq>);

  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Distance>, &stdex::is_lteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Distance>, &stdex::is_lt>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Distance>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<0>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_gteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_lt, applicability::permitted>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Dimensions<0>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Distance>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Distance>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Distance>, &stdex::is_gteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Distance>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<2>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<2>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<2>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_gteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<2>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<2>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<2>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_gteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<3>>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<>>, &stdex::is_gteq>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Dimensions<>>, stdex::ranges::repeat_view<Dimensions<>>, &stdex::is_neq, applicability::permitted>);

  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_lt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, stdex::ranges::repeat_view<Distance>, &stdex::is_neq>);

  // In theory, a boundless collection of Any<> could be Distance in the first element and Dimensions<0> otherwise.
  static_assert(compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<Distance, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_lt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, Distance, &stdex::is_neq>);

  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::tuple<Distance, Angle<>>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_lt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::tuple<Distance, Angle<>>, &stdex::is_neq>);

  static_assert(compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_eq>);
  static_assert(compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lteq>);
  static_assert(compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gteq>);
  static_assert(compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_lt>);
  static_assert(compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_gt>);
  static_assert(compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<std::vector<Distance>, stdex::ranges::repeat_view<Any<>>, &stdex::is_neq>);

  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_eq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_eq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_lteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_lteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_gteq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_gteq>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_lt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_gt, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_gt>);
  static_assert(compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_neq, applicability::permitted>);
  static_assert(not compares_with<stdex::ranges::repeat_view<Any<>>, std::vector<Distance>, &stdex::is_neq>);
}
