/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrix_tests.h"

using namespace OpenKalman;

TEST_F(matrix_tests, Coefficients_static) {
  // typedefs
  static_assert(Coefficients<Axis,Axis>::size == 2);
  static_assert(Coefficients<Axis,Axis>::dimension == 2);
  static_assert(Coefficients<Axis,Axis,Angle>::size == 3);
  static_assert(Coefficients<Axis,Axis,Angle>::dimension == 4);
  static_assert(Coefficients<Axis,Axis,Angle>::axes_only == false);
  static_assert(Coefficients<Angle,Axis,Axis>::axes_only == false);
  static_assert(Coefficients<Axis,Axis,Axis>::axes_only == true);

  // Prepend, Append
  static_assert(std::is_same_v<Coefficients<Angle,Axis>::Prepend<Axis>, Coefficients<Axis,Angle,Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, Angle>::Prepend<Axis>, Coefficients<Axis,Angle,Axis>>);
  static_assert(std::is_same_v<Coefficients<Angle,Axis>::Append<Angle>, Coefficients<Angle,Axis,Angle>>);
  static_assert(!std::is_same_v<Coefficients<Axis,Angle>::Append<Angle>, Coefficients<Axis,Angle,Axis>>);
  static_assert(std::is_same_v<Coefficients<Axis,Angle>::Append<Polar<Distance,Angle>>, Coefficients<Axis,Angle,Polar<Distance,Angle>>>);
  static_assert(std::is_same_v<Polar<Distance,Angle>::Append<Axis,Angle>, Coefficients<Polar<Distance,Angle>,Axis,Angle>>);
  static_assert(std::is_same_v<Polar<Angle,Distance>::Prepend<Axis,Angle>, Coefficients<Axis,Angle,Polar<Angle,Distance>>>);

  // Coefficient
  static_assert(std::is_same_v<Coefficients<Axis,Angle,Axis>::Coefficient<0>, Axis>);
  static_assert(std::is_same_v<Coefficients<Axis,Angle,Axis>::Coefficient<1>, Angle>);
  static_assert(std::is_same_v<Coefficients<Axis,Angle,Axis>::Coefficient<2>, Axis>);

  // Take
  static_assert(std::is_same_v<Coefficients<>::Take<0>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<Angle>::Take<1>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Coefficients<Axis,Angle,Axis,Axis,Angle>::Take<3>, Coefficients<Axis,Angle,Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis,Angle,Axis,Axis,Angle>::Take<3>, Coefficients<Axis,Angle,Axis,Axis>>);

  // Discard
  static_assert(std::is_same_v<Coefficients<Axis>::Discard<0>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Coefficients<Angle>::Discard<1>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<Axis,Angle,Axis,Axis,Angle>::Discard<3>, Coefficients<Axis,Angle>>);
  static_assert(std::is_same_v<Coefficients<Axis,Angle,Axis,Angle,Axis>::Discard<3>, Coefficients<Angle,Axis>>);

  // Replicate
  static_assert(std::is_same_v<Replicate<Angle, 0>, Coefficients<>>);
  static_assert(!std::is_same_v<Replicate<Angle, 0>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Replicate<Angle, 1>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Replicate<Angle, 2>, Coefficients<Angle, Angle>>);
  static_assert(std::is_same_v<Replicate<Coefficients<Angle, Axis>, 2>, Coefficients<Angle, Axis, Angle, Axis>>);
  static_assert(std::is_same_v<Axes<3>, Coefficients<Axis,Axis,Axis>>);

  // Concatenate
  static_assert(std::is_same_v<Concatenate<Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<>, Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Axis>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Coefficients<>>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Angle>, Axis>, Coefficients<Angle, Axis>>);
  static_assert(std::is_same_v<Concatenate<Axis, Coefficients<Angle>>, Coefficients<Axis, Angle>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<>, Coefficients<Angle>>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Coefficients<Angle>>, Coefficients<Axis, Angle>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis, Angle>, Coefficients<Angle, Axis>>, Coefficients<Axis, Angle, Angle, Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis, Angle>, Coefficients<Angle, Axis>, Coefficients<Axis, Angle>>,
    Coefficients<Axis, Angle, Angle, Axis, Axis, Angle>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Polar<Distance, Angle>>, Coefficients<Axis, Polar<Distance, Angle>>>);
  static_assert(std::is_same_v<Concatenate<Polar<Distance, Angle>, Coefficients<Axis>>, Coefficients<Polar<Distance, Angle>, Axis>>);
  static_assert(std::is_same_v<Concatenate<Polar<Distance, Angle>, Spherical<Distance, Angle, InclinationAngle>, Polar<Distance, Angle>>,
    Coefficients<Polar<Distance, Angle>, Spherical<Distance, Angle, InclinationAngle>, Polar<Distance, Angle>>>);

  // is_equivalent
  static_assert(is_equivalent_v<Coefficients<>, Coefficients<>>);
  static_assert(is_equivalent_v<Axis, Axis>);
  static_assert(not is_equivalent_v<Axis, Angle>);
  static_assert(not is_equivalent_v<Axis, Polar<>>);
  static_assert(is_equivalent_v<Axis, Coefficients<Axis>>);
  static_assert(is_equivalent_v<Coefficients<Axis>, Axis>);
  static_assert(is_equivalent_v<Coefficients<Axis>, Coefficients<Axis>>);
  static_assert(is_equivalent_v<Coefficients<Axis, Angle, Axis>, Coefficients<Axis, Angle, Axis>>);
  static_assert(is_equivalent_v<Coefficients<Coefficients<Axis>, Angle, Coefficients<Axis>>, Coefficients<Axis, Angle, Axis>>);
  static_assert(is_equivalent_v<Polar<Distance, Coefficients<Angle>>, Polar<Distance, Angle>>);
  static_assert(is_equivalent_v<Spherical<Distance, Coefficients<Angle>, InclinationAngle>, Spherical<Distance, Angle, InclinationAngle>>);
  static_assert(not is_equivalent_v<Coefficients<Axis, Angle, Angle>, Coefficients<Axis, Angle, Axis>>);
  static_assert(not is_equivalent_v<Coefficients<Axis, Angle>, Polar<Axis, Angle>>);

  // is_prefix
  static_assert(is_prefix_v<Coefficients<>, Axis>);
  static_assert(is_prefix_v<Coefficients<>, Coefficients<Axis>>);
  static_assert(is_prefix_v<Coefficients<>, Coefficients<Axis, Angle>>);
  static_assert(is_prefix_v<Coefficients<Axis>, Coefficients<Axis, Angle>>);
  static_assert(is_prefix_v<Axis, Coefficients<Axis, Angle>>);
  static_assert(not is_prefix_v<Angle, Coefficients<Axis, Angle>>);
  static_assert(not is_prefix_v<Coefficients<Angle>, Coefficients<Axis, Angle>>);
  static_assert(is_prefix_v<Coefficients<Axis, Angle>, Coefficients<Axis, Angle, Axis>>);
  static_assert(is_prefix_v<Coefficients<Axis, Angle, Axis>, Coefficients<Axis, Angle, Axis>>);
  static_assert(not is_prefix_v<Coefficients<Axis, Angle, Angle>, Coefficients<Axis, Angle, Axis>>);
}

