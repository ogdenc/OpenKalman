/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <gtest/gtest.h>

#include "basics/basics.hpp"
#include "coefficient-types/coefficient-types.hpp"

using namespace OpenKalman;

struct coefficients : public ::testing::Test
{
  coefficients() {}

  void SetUp() override {}

  void TearDown() override {}

  ~coefficients() override {}
};


TEST_F(coefficients, typedefs)
{
  static_assert(Coefficients<Axis, Axis>::size == 2);
  static_assert(Coefficients<Axis, Axis>::dimension == 2);
  static_assert(Coefficients<Axis, Axis, Angle>::size == 3);
  static_assert(Coefficients<Axis, Axis, Angle>::dimension == 4);
  static_assert(Coefficients<Axis, Axis, Angle>::axes_only == false);
  static_assert(Coefficients<Angle, Axis, Axis>::axes_only == false);
  static_assert(Coefficients<Axis, Axis, Axis>::axes_only == true);
}

TEST_F(coefficients, prepend_append)
{
  static_assert(std::is_same_v<Coefficients<Angle, Axis>::Prepend<Axis>, Coefficients<Axis, Angle, Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, Angle>::Prepend<Axis>, Coefficients<Axis, Angle, Axis>>);
  static_assert(std::is_same_v<Coefficients<Angle, Axis>::Append<Angle>, Coefficients<Angle, Axis, Angle>>);
  static_assert(!std::is_same_v<Coefficients<Axis, Angle>::Append<Angle>, Coefficients<Axis, Angle, Axis>>);
  static_assert(std::is_same_v<Coefficients<Axis, Angle>::Append<Polar<Distance, Angle>>, Coefficients<Axis, Angle, Polar<Distance, Angle>>>);
  static_assert(std::is_same_v<Polar<Distance, Angle>::Append<Axis, Angle>, Coefficients<Polar<Distance, Angle>, Axis, Angle>>);
  static_assert(std::is_same_v<Polar<Angle, Distance>::Prepend<Axis, Angle>, Coefficients<Axis, Angle, Polar<Angle, Distance>>>);
}

TEST_F(coefficients, Coefficients)
{
  static_assert(std::is_same_v<Coefficients<Axis, Angle, Axis>::Coefficient<0>, Axis>);
  static_assert(std::is_same_v<Coefficients<Axis, Angle, Axis>::Coefficient<1>, Angle>);
  static_assert(std::is_same_v<Coefficients<Axis, Angle, Axis>::Coefficient<2>, Axis>);
}

TEST_F(coefficients, Take)
{
  // Take
  static_assert(std::is_same_v<Coefficients<>::Take<0>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<Angle>::Take<1>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Coefficients<Axis, Angle, Axis, Axis, Angle>::Take<3>, Coefficients<Axis, Angle, Axis>>);
  static_assert(!std::is_same_v<Coefficients<Axis, Angle, Axis, Axis, Angle>::Take<3>, Coefficients<Axis, Angle, Axis, Axis>>);
}

TEST_F(coefficients, Discard)
{
  // Discard
  static_assert(std::is_same_v<Coefficients<Axis>::Discard<0>, Coefficients<Axis>>);
  static_assert(std::is_same_v<Coefficients<Angle>::Discard<1>, Coefficients<>>);
  static_assert(std::is_same_v<Coefficients<Axis, Angle, Axis, Axis, Angle>::Discard<3>, Coefficients<Axis, Angle>>);
  static_assert(std::is_same_v<Coefficients<Axis, Angle, Axis, Angle, Axis>::Discard<3>, Coefficients<Angle, Axis>>);
}

TEST_F(coefficients, Replicate)
{
  // Replicate
  static_assert(std::is_same_v<Replicate<Angle, 0>, Coefficients<>>);
  static_assert(!std::is_same_v<Replicate<Angle, 0>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Replicate<Angle, 1>, Coefficients<Angle>>);
  static_assert(std::is_same_v<Replicate<Angle, 2>, Coefficients<Angle, Angle>>);
  static_assert(std::is_same_v<Replicate<Coefficients<Angle, Axis>, 2>, Coefficients<Angle, Axis, Angle, Axis>>);
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
}

TEST_F(coefficients, equivalent_to)
{
  static_assert(equivalent_to<Coefficients<>, Coefficients<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(not equivalent_to<Axis, Angle>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, Coefficients<Axis>>);
  static_assert(equivalent_to<Coefficients<Axis>, Axis>);
  static_assert(equivalent_to<Coefficients<Axis>, Coefficients<Axis>>);
  static_assert(equivalent_to<Coefficients<Axis, Angle, Axis>, Coefficients<Axis, Angle, Axis>>);
  static_assert(equivalent_to<Coefficients<Coefficients<Axis>, Angle, Coefficients<Axis>>, Coefficients<Axis, Angle, Axis>>);
  static_assert(equivalent_to<Polar<Distance, Coefficients<Angle>>, Polar<Distance, Angle>>);
  static_assert(equivalent_to<Spherical<Distance, Coefficients<Angle>, InclinationAngle>, Spherical<Distance, Angle, InclinationAngle>>);
  static_assert(not equivalent_to<Coefficients<Axis, Angle, Angle>, Coefficients<Axis, Angle, Axis>>);
  static_assert(not equivalent_to<Coefficients<Axis, Angle>, Polar<Axis, Angle>>);
}

TEST_F(coefficients, prefix_of)
{
  static_assert(prefix_of<Coefficients<>, Axis>);
  static_assert(prefix_of<Coefficients<>, Coefficients<Axis>>);
  static_assert(prefix_of<Coefficients<>, Coefficients<Axis, Angle>>);
  static_assert(prefix_of<Coefficients<Axis>, Coefficients<Axis, Angle>>);
  static_assert(prefix_of<Axis, Coefficients<Axis, Angle>>);
  static_assert(not prefix_of<Angle, Coefficients<Axis, Angle>>);
  static_assert(not prefix_of<Coefficients<Angle>, Coefficients<Axis, Angle>>);
  static_assert(prefix_of<Coefficients<Axis, Angle>, Coefficients<Axis, Angle, Axis>>);
  static_assert(prefix_of<Coefficients<Axis, Angle, Axis>, Coefficients<Axis, Angle, Axis>>);
  static_assert(not prefix_of<Coefficients<Axis, Angle, Angle>, Coefficients<Axis, Angle, Axis>>);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
