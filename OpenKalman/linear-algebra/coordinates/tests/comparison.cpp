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
 * \brief Tests for \ref coordinate::compare_with
 */

#include "basics/tests/tests.hpp"
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

#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "basics/classes/equal_to.hpp"

TEST(coordinates, compares_with_equal_to)
{
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, unsigned, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, unsigned>);
  static_assert(compares_with<unsigned, std::integral_constant<std::size_t, 2>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<unsigned short, unsigned, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<unsigned short, unsigned>);

  static_assert(compares_with<std::integral_constant<std::size_t, 2>, unsigned, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<unsigned, std::integral_constant<std::size_t, 2>, equal_to<>, Applicability::permitted>);

  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<std::tuple<>, unsigned, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<unsigned, std::tuple<>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Axis, Dimensions<>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Dimensions<>, Axis, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<Axis, Polar<>, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<Polar<>, angle::Radians, equal_to<>, Applicability::permitted>);

  static_assert(compares_with<std::tuple<>, Dimensions<0>>);
  static_assert(compares_with<std::tuple<>, Dimensions<0>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<std::tuple<>, Dimensions<0>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Dimensions<0>, std::size_t, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<std::size_t, Dimensions<0>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Dimensions<1>, std::size_t, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Dimensions<>, Dimensions<10>, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<Dimensions<10>, Dimensions<5>, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, std::size_t, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<unsigned, angle::Radians, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<angle::Degrees, unsigned, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<std::tuple<Axis, angle::Radians, angle::Radians>, std::tuple<Axis, angle::Radians, inclination::Radians>, equal_to<>, Applicability::permitted>);
  static_assert(not compares_with<std::tuple<Axis, angle::Radians>, Polar<>, equal_to<>, Applicability::permitted>);

  static_assert(compares_with<Any<double>, Any<double>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Any<double>, Any<float>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Distance, Any<double>, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<Any<double>, unsigned, equal_to<>, Applicability::permitted>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, Any<double>, equal_to<>, Applicability::permitted>);

  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, unsigned int>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>>);
  static_assert(compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(not compares_with<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>);

  static_assert(compares_with<std::tuple<>, std::tuple<>>);
  static_assert(compares_with<Dimensions<0>, std::tuple<>>);
  static_assert(compares_with<std::tuple<Dimensions<0>>, std::tuple<>>);
  static_assert(compares_with<Axis, Axis>);
  static_assert(compares_with<Dimensions<1>, Axis>);
  static_assert(compares_with<std::tuple<Dimensions<1>>, std::tuple<Axis>>);
  static_assert(compares_with<Axis, Dimensions<1>>);
  static_assert(not compares_with<Axis, angle::Radians>);
  static_assert(not compares_with<angle::Degrees, angle::Radians>);
  static_assert(not compares_with<Axis, Polar<>>);
  static_assert(compares_with<Axis, std::tuple<Axis>>);
  static_assert(compares_with<std::tuple<Axis>, Axis>);
  static_assert(compares_with<std::tuple<Axis>, std::tuple<Axis>>);
  static_assert(compares_with<std::tuple<Axis, angle::Radians, Axis>, std::tuple<Axis, angle::Radians, Axis>>);
  static_assert(compares_with<std::tuple<Dimensions<2>, inclination::Radians, Dimensions<3>>, std::tuple<Axis, Axis, inclination::Radians, Axis, Axis, Axis>>);
  static_assert(compares_with<Polar<>, Polar<>>);
  static_assert(compares_with<Spherical<>, Spherical<>>);

  static_assert(not compares_with<Dimensions<0>, std::size_t>);
  static_assert(not compares_with<Dimensions<1>, std::size_t>);
  static_assert(not compares_with<Dimensions<>, Dimensions<10>>);
  static_assert(not compares_with<Dimensions<>, unsigned>);
  static_assert(not compares_with<angle::Degrees, Dimensions<1>>);
  static_assert(not compares_with<angle::Degrees, std::size_t>);
  static_assert(not compares_with<unsigned, angle::Radians>);
  static_assert(not compares_with<angle::Degrees, unsigned>);
}

#include "basics/classes/less_equal.hpp"

TEST(coordinates, compares_with_less)
{
  using namespace coordinate::internal;
  static_assert(compares_with<std::tuple<>, Axis, less_equal<>>);
  static_assert(compares_with<std::tuple<>, Dimensions<2>, less_equal<>>);
  static_assert(compares_with<std::tuple<>, std::tuple<Axis>, less_equal<>>);
  static_assert(compares_with<std::tuple<>, std::tuple<Axis, angle::Radians>, less_equal<>>);
  static_assert(compares_with<std::tuple<Axis, angle::Radians>, std::tuple<Axis, angle::Radians>, less_equal<>>);
  static_assert(compares_with<std::tuple<Axis>, std::tuple<Axis, angle::Radians>, less_equal<>>);
  static_assert(compares_with<std::tuple<Axis>, std::tuple<Dimensions<2>, angle::Radians>, less_equal<>>);
  static_assert(compares_with<Axis, std::tuple<Axis, angle::Radians>, less_equal<>>);
  static_assert(compares_with<Axis, std::tuple<Dimensions<2>, angle::Radians>, less_equal<>>);
  static_assert(not compares_with<angle::Radians, std::tuple<Axis, angle::Radians>, less_equal<>>);
  static_assert(not compares_with<std::tuple<angle::Radians>, std::tuple<Axis, angle::Radians>, less_equal<>>);
  static_assert(compares_with<std::tuple<Axis, angle::Radians>, std::tuple<Axis, angle::Radians, Axis>, less_equal<>>);
  static_assert(compares_with<std::tuple<Axis, angle::Radians, Axis>, std::tuple<Axis, angle::Radians, Axis>, less_equal<>>);
  static_assert(not compares_with<std::tuple<Axis, angle::Radians, angle::Radians>, std::tuple<Axis, angle::Radians, Axis>, less_equal<>>);
  static_assert(not compares_with<Dimensions<>, Axis, less_equal<>>);
}
