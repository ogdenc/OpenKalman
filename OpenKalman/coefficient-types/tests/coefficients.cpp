/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for coefficient types
 */

#include "coefficients.hpp"


TEST_F(coefficients, typedefs)
{
  static_assert(Coefficients<Axis, Axis>::size == 2);
  static_assert(Coefficients<Axis, Axis>::dimension == 2);
  static_assert(Coefficients<Axis, Axis, angle::Radians>::size == 3);
  static_assert(Coefficients<Axis, Axis, angle::Radians>::dimension == 4);
  static_assert(Coefficients<Axis, Axis, angle::Radians>::axes_only == false);
  static_assert(Coefficients<angle::Radians, Axis, Axis>::axes_only == false);
  static_assert(Coefficients<Axis, Axis, Axis>::axes_only == true);
}


TEST_F(coefficients, prepend_append)
{
  static_assert(std::is_same_v<Coefficients<angle::Radians, Axis>::Prepend<Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, angle::Radians>::Prepend<Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Coefficients<angle::Radians, Axis>::Append<angle::Radians>, Coefficients<angle::Radians, Axis, angle::Radians>>);
  static_assert(!std::is_same_v<Coefficients<Axis, angle::Radians>::Append<angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians>::Append<Polar<Distance, angle::Radians>>, Coefficients<Axis, angle::Radians, Polar<Distance, angle::Radians>>>);
}


TEST_F(coefficients, Coefficients)
{
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis>::Coefficient<0>, Axis>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis>::Coefficient<1>, angle::Radians>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis>::Coefficient<2>, Axis>);
}


TEST_F(coefficients, Take)
{
  // Take
  static_assert(std::is_same_v<Coefficients<>::Take<0>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<angle::Radians>::Take<1>, Coefficients<angle::Radians>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, Coefficients<Axis, angle::Radians, Axis, Axis>>);
}


TEST_F(coefficients, Discard)
{
  // Discard
  static_assert(std::is_same_v<Coefficients<Axis>::Discard<0>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Coefficients<angle::Radians>::Discard<1>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis, Axis, angle::Radians>::Discard<3>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<Coefficients<Axis, angle::Radians, Axis, angle::Radians, Axis>::Discard<3>, Coefficients<angle::Radians, Axis>>);
}


TEST_F(coefficients, Replicate)
{
  // Replicate
  static_assert(std::is_same_v<Replicate<angle::Radians, 0>, Coefficients<>>);
  static_assert(!std::is_same_v<Replicate<angle::Radians, 0>, Coefficients<angle::Radians>>);
  static_assert(std::is_same_v<Replicate<angle::Radians, 1>, Coefficients<angle::Radians>>);
  static_assert(std::is_same_v<Replicate<angle::Radians, 2>, Coefficients<angle::Radians, angle::Radians>>);
  static_assert(std::is_same_v<Replicate<Coefficients<angle::Radians, Axis>, 2>, Coefficients<angle::Radians, Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Axes<3>, Coefficients<Axis, Axis, Axis>>);
}


TEST_F(coefficients, Concatenate)
{
  // Concatenate
  static_assert(std::is_same_v<Concatenate<Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<>, Coefficients<>>, Coefficients<>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Axis>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Coefficients<>>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<angle::Radians>, Axis>, Coefficients<angle::Radians, Axis>>);
  static_assert(std::is_same_v<Concatenate<Axis, Coefficients<angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<>, Coefficients<angle::Radians>>, Coefficients<angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Coefficients<angle::Radians>>, Coefficients<Axis, angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis, angle::Radians>, Coefficients<angle::Radians, Axis>>, Coefficients<Axis, angle::Radians, angle::Radians, Axis>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis, angle::Radians>, Coefficients<angle::Radians, Axis>, Coefficients<Axis, angle::Radians>>,
    Coefficients<Axis, angle::Radians, angle::Radians, Axis, Axis, angle::Radians>>);
  static_assert(std::is_same_v<Concatenate<Coefficients<Axis>, Polar<Distance, angle::Radians>>, Coefficients<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<Concatenate<Polar<Distance, angle::Radians>, Coefficients<Axis>>, Coefficients<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<Concatenate<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>,
    Coefficients<Polar<Distance, angle::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>, Polar<Distance, angle::Radians>>>);
}


TEST_F(coefficients, equivalent_to)
{
  static_assert(equivalent_to<Coefficients<>, Coefficients<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, Coefficients<Axis>>);
  static_assert(equivalent_to<Coefficients<Axis>, Axis>);
  static_assert(equivalent_to<Coefficients<Axis>, Coefficients<Axis>>);
  static_assert(equivalent_to<Coefficients<Axis, angle::Radians, Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Coefficients<Coefficients<Axis>, angle::Radians, Coefficients<Axis>>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<Distance, Coefficients<angle::Radians>>, Polar<Distance, angle::Radians>>);
  static_assert(equivalent_to<Spherical<Distance, Coefficients<angle::Radians>, inclination::Radians>, Spherical<Distance, angle::Radians, inclination::Radians>>);
  static_assert(not equivalent_to<Coefficients<Axis, angle::Radians, angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(not equivalent_to<Coefficients<Axis, angle::Radians>, Polar<Axis, angle::Radians>>);
}


TEST_F(coefficients, prefix_of)
{
  static_assert(prefix_of<Coefficients<>, Axis>);
  static_assert(prefix_of<Coefficients<>, Coefficients<Axis>>);
  static_assert(prefix_of<Coefficients<>, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Coefficients<Axis>, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Axis, Coefficients<Axis, angle::Radians>>);
  static_assert(not prefix_of<angle::Radians, Coefficients<Axis, angle::Radians>>);
  static_assert(not prefix_of<Coefficients<angle::Radians>, Coefficients<Axis, angle::Radians>>);
  static_assert(prefix_of<Coefficients<Axis, angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(prefix_of<Coefficients<Axis, angle::Radians, Axis>, Coefficients<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<Coefficients<Axis, angle::Radians, angle::Radians>, Coefficients<Axis, angle::Radians, Axis>>);
}


int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
