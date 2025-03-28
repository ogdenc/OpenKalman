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
 * \brief Tests for \ref coordinate::pattern equivalence
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
#include "linear-algebra/coordinates/views/comparison.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinate;
using coordinate::views::comparison;

#include "linear-algebra/coordinates/functions/comparison-operators.hpp"

TEST(coordinates, compare_fixed_pattern)
{
  static_assert(Dimensions<3>{} == std::tuple<Axis, Axis, Axis>{});
  static_assert(Dimensions<3>{} <= std::tuple<Axis, Axis, Axis>{});
  static_assert(Dimensions<3>{} <= std::tuple<Axis, Axis, Axis, Axis>{});
  static_assert(Dimensions<3>{} < std::tuple<Axis, Axis, Axis, Axis>{});
  static_assert(Dimensions<4>{} >= std::tuple<Axis, Axis, Axis, Axis>{});
  static_assert(Dimensions<4>{} >= std::tuple<Axis, Axis, Axis>{});
  static_assert(Dimensions<4>{} > std::tuple<Axis, Axis, Axis>{});

  static_assert(std::tuple<Axis, Axis, Axis>{} == Dimensions<3>{});
  static_assert(std::tuple<Axis, Axis, Axis>{} >= Dimensions<3>{});
  static_assert(std::tuple<Axis, Axis, Axis, Axis>{} >= Dimensions<3>{});
  static_assert(std::tuple<Axis, Axis, Axis, Axis>{} > Dimensions<3>{});
  static_assert(std::tuple<Axis, Axis, Axis, Axis>{} <= Dimensions<4>{});
  static_assert(std::tuple<Axis, Axis, Axis>{} <= Dimensions<4>{});
  static_assert(std::tuple<Axis, Axis, Axis>{} < Dimensions<4>{});

  static_assert(std::tuple<>{} == std::tuple<>{});
  static_assert(std::tuple<Axis, angle::Radians>{} == std::tuple<Axis, angle::Radians>{});
  static_assert(std::tuple<Axis, angle::Radians>{} <= std::tuple<Axis, angle::Radians>{});
  static_assert(std::tuple<Axis, angle::Radians>{} >= std::tuple<Axis, angle::Radians>{});

  static_assert(comparison(std::tuple<>{}) <= std::tuple<Axis, angle::Radians>{});
  static_assert(comparison(std::tuple<Axis>{}) <= std::tuple<Axis, angle::Radians>{});
  static_assert(comparison(std::tuple<Axis, angle::Radians>{}) < std::tuple<Axis, angle::Radians, Axis>{});
  static_assert(std::tuple<Axis, angle::Radians, Axis>{} > comparison(std::tuple<Axis, angle::Radians>{}));
  static_assert(comparison(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}) <= std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{});
  static_assert(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{} == comparison(std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(comparison(std::tuple<Axis, Dimensions<3>, angle::Radians, Axis, Dimensions<2>>{}) < std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{});
}


TEST(coordinates, compare_dynamic_pattern)
{
  static_assert(Dimensions{0} == std::tuple<>{});
  static_assert(std::tuple<>{} == Dimensions{0});

  static_assert(Dimensions{3} == std::tuple<Axis, Axis, Axis>{});
  static_assert(Dimensions{4} == std::tuple<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= std::tuple<Axis, Dimensions<2>>{});
  static_assert(Dimensions{3} < std::tuple<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} != std::tuple<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= std::tuple<Axis, Axis, Axis, Axis>{});

  static_assert(std::tuple<Axis, Axis, Axis>{} == Dimensions{3});
  static_assert(std::tuple<Axis, Dimensions<2>, Axis>{} == Dimensions{4});
  static_assert(std::tuple<Axis, Dimensions<2>>{} >= Dimensions{3});
  static_assert(std::tuple<Axis, Dimensions<2>, Axis>{} > Dimensions{3});
  static_assert(std::tuple<Axis, Dimensions<2>, Axis>{} != Dimensions{3});
  static_assert(std::tuple<Axis, Axis, Axis, Axis>{} >= Dimensions{3});

  // delete these after debugging:
  std::cout << "-------------" << std::endl;
  EXPECT_TRUE((comparison(make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5>{})) == comparison(make_pattern_vector(Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}))));
  std::cout << "-------------" << std::endl;
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<4>{}) < comparison(make_pattern_vector(Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}))));
  std::cout << "-------------" << std::endl;


  EXPECT_TRUE(Any<double>{Axis{}} == Any<double>{Axis{}});
  EXPECT_TRUE(Any<double>{Dimensions<3>{}} == Any<double>{Dimensions<3>{}});
  EXPECT_TRUE(Any<double>{Dimensions<4>{}} != Any<double>{Dimensions<3>{}});
  EXPECT_TRUE(Any<double>{Dimensions{4}} == Any<double>{Dimensions{4}});
  EXPECT_TRUE(Any<double>{Dimensions{4}} != Any<double>{Dimensions{5}});
  EXPECT_TRUE(Any<double>{Dimensions<2>{}} == Dimensions<2>{});
  EXPECT_TRUE(Any<double>{Dimensions<2>{}} == Dimensions{2});
  EXPECT_TRUE(Any<double>{Dimensions<2>{}} < Dimensions{3});
  EXPECT_TRUE(Dimensions{3} == Any<double>{Dimensions<3>{}});
  EXPECT_TRUE(Dimensions{2} < Any<double>{Dimensions<3>{}});

  EXPECT_TRUE(Any<double>{angle::Radians{}} == Any<double>{angle::Radians{}});
  EXPECT_TRUE(Any<double>{angle::Radians{}} != Any<double>{angle::Degrees{}});
  EXPECT_TRUE(Any<double>{angle::Radians{}} == angle::Radians{});
  EXPECT_TRUE(angle::Radians{} == Any<double>{angle::Radians{}});
  EXPECT_TRUE(angle::Degrees{} != Any<double>{angle::Radians{}});


  EXPECT_TRUE((make_pattern_vector() == make_pattern_vector()));
  EXPECT_TRUE((make_pattern_vector(Axis{}) == make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) <= make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) == make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) < make_pattern_vector(Axis{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) == make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Axis{}) > make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) == make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Axis{}) >= make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) != make_pattern_vector(angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) != make_pattern_vector(Polar<>{})));

  EXPECT_TRUE((make_pattern_vector() < make_pattern_vector(angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(angle::Radians{}) > make_pattern_vector()));
  EXPECT_TRUE((make_pattern_vector(angle::Radians{}) == make_pattern_vector(angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(angle::Degrees{}) == make_pattern_vector(Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{})));
  EXPECT_TRUE((make_pattern_vector(angle::Degrees{}) == make_pattern_vector(Angle<value::Fixed<double, -180>, value::Fixed<long double, 180>>{})));
  EXPECT_TRUE((make_pattern_vector(angle::PositiveDegrees{}) == make_pattern_vector(Angle<std::integral_constant<int, 0>, std::integral_constant<std::size_t, 360>>{})));
  EXPECT_TRUE((make_pattern_vector(angle::Degrees{}) != make_pattern_vector(angle::PositiveDegrees{})));
  EXPECT_TRUE((make_pattern_vector(angle::Degrees{}) != make_pattern_vector(angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(inclination::Radians{}) != make_pattern_vector(inclination::Degrees{})));

  EXPECT_TRUE((make_pattern_vector() < make_pattern_vector(inclination::Radians{})));
  EXPECT_TRUE((make_pattern_vector(inclination::Radians{}) > make_pattern_vector()));
  EXPECT_TRUE((make_pattern_vector(inclination::Radians{}) == make_pattern_vector(inclination::Radians{})));
  EXPECT_TRUE((make_pattern_vector(inclination::Degrees{}) == make_pattern_vector(Inclination<std::integral_constant<int, -90>, std::integral_constant<int, 90>>{})));
  EXPECT_TRUE((make_pattern_vector(inclination::Degrees{}) == make_pattern_vector(Inclination<value::Fixed<double, -90>, value::Fixed<long double, 90>>{})));

  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Axis{}) == make_pattern_vector(Axis{}, angle::Radians{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Axis{}) < make_pattern_vector(Axis{}, angle::Radians{}, Axis{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Axis{}, Axis{}) > make_pattern_vector(Axis{}, angle::Radians{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Axis{}) == make_pattern_vector(Axis{}, angle::Radians{}, Axis{})));
  EXPECT_TRUE((comparison(make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5>{})) == make_pattern_vector(Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Axis{}) == make_pattern_vector(Axis{}, angle::Radians{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<4>{}) < comparison(make_pattern_vector(Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}))));
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5>{}) > make_pattern_vector(Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<2>{})));
  EXPECT_TRUE((make_pattern_vector(Polar<Distance, angle::Radians>{}) == make_pattern_vector(Polar<Distance, angle::Radians>{})));
  EXPECT_TRUE((make_pattern_vector(Polar<Distance, angle::Radians>{}, Axis{}) == make_pattern_vector(Polar<Distance, angle::Radians>{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Polar<Distance, angle::Radians>{}, Axis{}) != make_pattern_vector(Polar<Distance, angle::Radians>{})));
  EXPECT_TRUE((make_pattern_vector(Polar<Distance, angle::Radians>{}, Axis{}) > make_pattern_vector(Polar<Distance, angle::Radians>{})));
  EXPECT_TRUE((make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{}) == make_pattern_vector(Spherical<Distance, angle::Radians, inclination::Radians>{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, angle::Radians{}) != make_pattern_vector(Axis{}, angle::Radians{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}) != make_pattern_vector(Polar<Distance, angle::Radians>{})));

  EXPECT_TRUE((make_pattern_vector(Axis{}) == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) == Dimensions<1>{}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Axis{}, Axis{}) == Dimensions<3>{}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Axis{}) < Dimensions<3>{}));
  EXPECT_TRUE((std::tuple<Axis, Axis>{} < make_pattern_vector(Dimensions<4>{})));
  EXPECT_TRUE((make_pattern_vector(Dimensions<4>{}) > std::tuple<Axis, Axis>{}));
  EXPECT_TRUE((make_pattern_vector(Dimensions<4>{}) != std::tuple<Axis, Axis>{}));
  EXPECT_TRUE((std::tuple<>{} == make_pattern_vector()));

  EXPECT_TRUE((make_pattern_vector(Axis{}) == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == make_pattern_vector(Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}) == Dimensions{1}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Axis{}, Axis{}) == Dimensions{3}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Axis{}) < Dimensions{4}));
  EXPECT_TRUE((std::tuple<Axis, Axis>{} < make_pattern_vector(Dimensions{4})));
  EXPECT_TRUE((make_pattern_vector(Dimensions{4}) > std::tuple<Axis, Axis>{}));
  EXPECT_TRUE((make_pattern_vector(Dimensions{2}) < std::tuple<Axis, Axis, Axis, Axis>{}));
  EXPECT_TRUE((Dimensions{4} != make_pattern_vector(Axis{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(angle::Radians{}) == angle::Radians{}));
  EXPECT_TRUE((make_pattern_vector(angle::Degrees{}) == std::tuple<angle::Degrees>{}));
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}, angle::Degrees{}) == std::tuple<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((angle::Radians{} == make_pattern_vector(angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}, angle::Degrees{}) > Dimensions<3>{}));
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}) < std::tuple<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((make_pattern_vector(Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}) == std::tuple<Dimensions<3>, angle::Degrees, Dimensions<5>>{}));
  EXPECT_TRUE((std::tuple<Dimensions<3>, angle::Degrees, Dimensions<5>>{} == make_pattern_vector(Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}) == std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}) == std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}) < std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}) > std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}));
  EXPECT_TRUE((make_pattern_vector(Axis{}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}) != std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_FALSE((make_pattern_vector(Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}) < std::tuple<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}));

  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}) == std::tuple<Axis, angle::Radians>{}));
  EXPECT_TRUE((std::tuple<Axis, angle::Radians>{} <= make_pattern_vector(Axis{}, angle::Radians{})));
  EXPECT_TRUE((std::tuple<Axis, angle::Radians>{} <= make_pattern_vector(Axis{}, angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}) >= std::tuple<Axis, angle::Radians>{}));
  EXPECT_TRUE((std::tuple<Axis, angle::Radians>{} < make_pattern_vector(Axis{}, angle::Radians{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}) <= std::tuple<Axis, angle::Radians, Axis>{}));
  EXPECT_TRUE((angle::Radians{} == make_pattern_vector(angle::Radians{})));
  EXPECT_TRUE((make_pattern_vector(inclination::Radians{}) == inclination::Radians{}));
  EXPECT_TRUE((angle::Radians{} != make_pattern_vector(inclination::Radians{})));
  EXPECT_FALSE((angle::Radians{} < inclination::Radians{}));
  EXPECT_TRUE((make_pattern_vector(Polar<Distance, angle::Radians>{}) != Dimensions<5>{}));
  EXPECT_FALSE((Polar<Distance, angle::Radians>{} < make_pattern_vector(Dimensions<5>{})));
  EXPECT_TRUE((make_pattern_vector(Spherical<Distance, inclination::Radians, angle::Radians>{}) != Dimensions<5>{}));
  EXPECT_FALSE((Spherical<Distance, inclination::Radians, angle::Radians>{} < make_pattern_vector(Dimensions<5>{})));

  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}) == make_pattern_vector<float> (Axis{}, angle::Radians{}, Distance{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}) == make_pattern_vector<long double> (Axis{}, angle::Radians{}, Distance{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}) < make_pattern_vector<float> (Axis{}, angle::Radians{}, Distance{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}, Axis{}) > make_pattern_vector<float> (Axis{}, angle::Radians{}, Distance{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}) <= make_pattern_vector<float> (Axis{}, angle::Radians{}, Distance{}, Axis{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}, Axis{}) >= make_pattern_vector<float> (Axis{}, angle::Radians{}, Distance{})));
  EXPECT_TRUE((make_pattern_vector(Axis{}, angle::Radians{}, Distance{}, Axis{}) != make_pattern_vector<float> (Axis{}, angle::Radians{}, Distance{})));
}
