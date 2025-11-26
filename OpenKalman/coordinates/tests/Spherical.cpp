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
 * \brief Tests for coordinates::Spherical
 */

#include "collections/tests/tests.hpp"
#include "coordinates/concepts/fixed_pattern.hpp"
#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/euclidean_pattern.hpp"
#include "coordinates/concepts/descriptor.hpp"
#include "coordinates/functions/get_stat_dimension.hpp"
#include "coordinates/functions/get_is_euclidean.hpp"
#include "coordinates/traits/dimension_of.hpp"
#include "coordinates/traits/stat_dimension_of.hpp"

#include "coordinates/descriptors/Spherical.hpp"
#include "coordinates/descriptors/Distance.hpp"
#include "coordinates/descriptors/Angle.hpp"
#include "coordinates/descriptors/Inclination.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

using OpenKalman::stdex::numbers::pi;

TEST(coordinates, Spherical)
{
  static_assert(descriptor<Spherical<>>);
  static_assert(fixed_pattern<Spherical<>>);
  static_assert(pattern<Spherical<Distance, angle::Degrees, inclination::Degrees>>);
  static_assert(not euclidean_pattern<Spherical<>>);

  static_assert(get_dimension(Spherical<>{}) == 3);
  static_assert(get_stat_dimension(Spherical<>{}) == 4);
  static_assert(not get_is_euclidean(Spherical<>{}));
  static_assert(dimension_of_v<Spherical<>> == 3);
  static_assert(stat_dimension_of_v<Spherical<>> == 4);

  static_assert(std::is_same_v<std::common_type_t<Spherical<Distance, inclination::Degrees, angle::Degrees>,
    Spherical<Distance, Inclination<std::integral_constant<int, 180>>, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>>>,
    Spherical<Distance, inclination::Degrees, angle::Degrees>>);
}

#include "coordinates/functions/to_stat_space.hpp"

namespace
{
  constexpr double sqrt3_4 = values::sqrt(3.)/4;
}

TEST(coordinates, Spherical_to_stat_space)
{
  using S1 = Spherical<Distance, inclination::Radians, angle::Radians>;
  using S2 = Spherical<Distance, inclination::Radians, angle::Circle>;
  using S3 = Spherical<Distance, angle::Radians, inclination::Radians>;
  using S4 = Spherical<angle::PositiveRadians, Distance, inclination::Radians>;
  using S5 = Spherical<inclination::Radians, Distance, angle::PositiveDegrees>;
  using S6 = Spherical<angle::Degrees, inclination::Radians, Distance>;
  using S7 = Spherical<inclination::Degrees, angle::Radians, Distance>;
  using S8 = Spherical<inclination::Degrees, angle::PositiveDegrees, Distance>;

  static_assert(values::internal::near(to_stat_space(S1{}, std::array{2., pi/3, pi/6})[1U], 0.75, 1e-6));
  static_assert(values::internal::near(to_stat_space(S1{}, std::array{2., pi/3, pi/6})[2U], sqrt3_4, 1e-6));
  static_assert(values::internal::near(to_stat_space(S1{}, std::array{2., pi/3, pi/6})[3U], 0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(S5{}, std::array{pi/3, 2., 390.})[1U], 0.75, 1e-6));
  static_assert(values::internal::near(to_stat_space(S5{}, std::array{pi/3, 2., 390.})[2U], sqrt3_4, 1e-6));
  static_assert(values::internal::near(to_stat_space(S5{}, std::array{pi/3, 2., 390.})[3U], 0.5, 1e-6));

  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<1>(to_stat_space(S8{}, std::tuple{values::fixed_value<double, -240>{}, values::fixed_value<double, 60>{}, values::fixed_value<double, 2>{}})))>, sqrt3_4, 1e-6));
  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<2>(to_stat_space(S8{}, std::tuple{values::fixed_value<double, -240>{}, values::fixed_value<double, 60>{}, values::fixed_value<double, 2>{}})))>, .75, 1e-6));
  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<3>(to_stat_space(S8{}, std::tuple{values::fixed_value<double, -240>{}, values::fixed_value<double, 60>{}, values::fixed_value<double, 2>{}})))>, -0.5, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{2., pi/3, pi/6}), std::array{2., 0.75, sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{2., pi/3, -pi/6}), std::array{2., 0.75, -sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{2., -pi/3, pi/6}), std::array{2., -0.75, -sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{2., -pi/3, -pi/6}), std::array{2., -0.75, sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{-2., pi/3, pi/6}), std::array{2., -0.75, -sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{-2., pi/3, -pi/6}), std::array{2., -0.75, sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{-2., -pi/3, pi/6}), std::array{2., 0.75, sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{-2., -pi/3, -pi/6}), std::array{2., 0.75, -sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S1{}, std::vector{-2., -4*pi/3, -7*pi/6}), std::array{2., 0.75, -sqrt3_4, 0.5}, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(S2{}, std::array{-2., pi/3, 1./12}), std::array{2., -0.75, -sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S3{}, std::tuple{2, 7*pi/6, pi/3}), std::array{2., -0.75, -sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S4{}, std::vector{7*pi/6, 2., pi/3}), std::array{2., -0.75, -sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S5{}, std::array{pi/3, 2., 390.}), std::array{2., 0.75, sqrt3_4, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S6{}, std::tuple{30, 4*pi/3, 2}), std::array{2., -0.75, -sqrt3_4, -0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(S7{}, std::vector{-240., pi/6, 2.}), std::array{2., 0.75, sqrt3_4, -0.5}, 1e-6));
}

#include "coordinates/functions/from_stat_space.hpp"

TEST(coordinates, Spherical_from_stat_space)
{
  using S1 = Spherical<Distance, inclination::Radians, angle::Radians>;
  using S2 = Spherical<Distance, inclination::Radians, angle::Circle>;
  using S3 = Spherical<Distance, angle::Radians, inclination::Radians>;
  using S4 = Spherical<angle::PositiveRadians, Distance, inclination::Radians>;
  using S5 = Spherical<inclination::Radians, Distance, angle::PositiveDegrees>;
  using S6 = Spherical<angle::Degrees, inclination::Radians, Distance>;
  using S7 = Spherical<inclination::Degrees, angle::Radians, Distance>;
  using S8 = Spherical<inclination::Degrees, angle::PositiveDegrees, Distance>;

  static_assert(values::internal::near(from_stat_space(S1{}, std::array{2., 0.75, -sqrt3_4, -0.5})[0U], 2., 1e-6));
  static_assert(values::internal::near(from_stat_space(S1{}, std::array{2., 0.75, -sqrt3_4, -0.5})[1U], 2*pi/3, 1e-6));
  static_assert(values::internal::near(from_stat_space(S1{}, std::array{2., 0.75, -sqrt3_4, -0.5})[2U], -pi/6, 1e-6));
  static_assert(values::internal::near(from_stat_space(S7{}, std::array{2., 0.75, sqrt3_4, -0.5})[0U], 120., 1e-6));
  static_assert(values::internal::near(from_stat_space(S7{}, std::array{2., 0.75, sqrt3_4, -0.5})[1U], pi/6, 1e-6));
  static_assert(values::internal::near(from_stat_space(S7{}, std::array{2., 0.75, sqrt3_4, -0.5})[2U], 2., 1e-6));

  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<1>(from_stat_space(S8{}, std::tuple{values::fixed_value<double, 2>{}, values::fixed_value<double, 1>{}, values::fixed_value<double, -1>{}, values::fixed_value<double, -1>{}})))>, 315, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., 0.75, sqrt3_4, 0.5}), std::array{2., pi/3, pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., 0.75, -sqrt3_4, 0.5}), std::array{2., pi/3, -pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., -0.75, -sqrt3_4, 0.5}), std::array{2., pi/3, -5*pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., -0.75, sqrt3_4, 0.5}), std::array{2., pi/3, 5*pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., -0.75, -sqrt3_4, -0.5}), std::array{2., 2*pi/3, -5*pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., -0.75, sqrt3_4, -0.5}), std::array{2., 2*pi/3, 5*pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., 0.75, sqrt3_4, -0.5}), std::array{2., 2*pi/3, pi/6}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S1{}, std::vector{2., 0.75, -sqrt3_4, -0.5}), std::array{2., 2*pi/3, -pi/6}, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(S2{}, std::array{2., -0.75, -sqrt3_4, -0.5}), std::array{2., 2*pi/3, 7./12}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S3{}, std::tuple{2, -0.75, -sqrt3_4, 0.5}), std::array{2., -5*pi/6, pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S4{}, std::vector{2., -0.75, -sqrt3_4, 0.5}), std::array{7*pi/6, 2., pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S5{}, std::array{2., 0.75, sqrt3_4, 0.5}), std::array{pi/3, 2., 30.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S6{}, std::tuple{2, -0.75, -sqrt3_4, -0.5}), std::array{-150., 2*pi/3, 2.}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(S7{}, std::vector{2., 0.75, sqrt3_4, -0.5}), std::array{120., pi/6, 2.}, 1e-6));
}

#include "coordinates/functions/wrap.hpp"

TEST(coordinates, Spherical_wrap)
{
  constexpr auto cmp = [](const auto& c, const auto& v){ return test::is_near(wrap(c, v), from_stat_space(c, to_stat_space(c, v)), 1e-6); };

  using S1 = Spherical<Distance, inclination::Radians, angle::Radians>;
  using S2 = Spherical<Distance, inclination::Radians, angle::Circle>;
  using S3 = Spherical<Distance, angle::Radians, inclination::Radians>;
  using S4 = Spherical<angle::PositiveRadians, Distance, inclination::Radians>;
  using S5 = Spherical<inclination::Radians, Distance, angle::PositiveDegrees>;
  using S6 = Spherical<angle::Degrees, inclination::Radians, Distance>;
  using S7 = Spherical<inclination::Degrees, angle::Radians, Distance>;
  using S8 = Spherical<inclination::Degrees, angle::PositiveDegrees, Distance>;

  static_assert(values::internal::near(wrap(S2{}, std::array{-2., pi/3, 1./12})[0U], 2., 1e-6));
  static_assert(values::internal::near(wrap(S2{}, std::array{-2., pi/3, 1./12})[1U], 2*pi/3, 1e-6));
  static_assert(values::internal::near(wrap(S2{}, std::array{-2., pi/3, 1./12})[2U], 7./12, 1e-6));

  static_assert(values::internal::near(wrap(S6{}, std::array{30., 4*pi/3, 2.})[0U], -150., 1e-6));
  static_assert(values::internal::near(wrap(S6{}, std::array{30., 4*pi/3, 2.})[1U], 2*pi/3, 1e-6));
  static_assert(values::internal::near(wrap(S6{}, std::array{30., 4*pi/3, 2.})[2U], 2., 1e-6));

  static_assert(values::internal::near(wrap(S8{}, std::array{-240., 60., -2.})[0U], 60., 1e-6));
  static_assert(values::internal::near(wrap(S8{}, std::array{-240., 60., -2.})[1U], 240., 1e-6));
  static_assert(values::internal::near(wrap(S8{}, std::array{-240., 60., -2.})[2U], 2., 1e-6));

  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<0>(wrap(S8{}, std::tuple{values::fixed_value<double, -240>{}, values::fixed_value<double, 60>{}, values::fixed_value<double, -2>{}})))>, 60., 1e-6));
  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<1>(wrap(S8{}, std::tuple{values::fixed_value<double, -240>{}, values::fixed_value<double, 60>{}, values::fixed_value<double, -2>{}})))>, 240, 1e-6));
  static_assert(values::internal::near(values::fixed_value_of_v<decltype(std::get<2>(wrap(S8{}, std::tuple{values::fixed_value<double, -240>{}, values::fixed_value<double, 60>{}, values::fixed_value<double, -2>{}})))>, 2., 1e-6));

  EXPECT_TRUE(cmp(S1{}, std::vector{2., pi/3, pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{2., pi/3, -pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{2., -pi/3, pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{2., -pi/3, -pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{-2., pi/3, pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{-2., -pi/3, pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{-2., pi/3, -pi/6}));
  EXPECT_TRUE(cmp(S1{}, std::vector{-2., -pi/3, -pi/6}));

  EXPECT_TRUE(test::is_near(wrap(S2{}, std::array{-2., pi/3, 1./12}), std::array{2., 2*pi/3, 7./12}, 1e-6));
  EXPECT_TRUE(cmp(S2{}, std::array{-2., pi/3, 1./12}));

  EXPECT_TRUE(cmp(S3{}, std::tuple{2., 7*pi/6, pi/3}));
  EXPECT_TRUE(cmp(S4{}, std::vector{7*pi/6, 2., pi/3}));
  EXPECT_TRUE(cmp(S5{}, std::array{pi/3, 2., 390.}));

  EXPECT_TRUE(test::is_near(wrap(S6{}, std::array{30., 4*pi/3, 2.}), std::array{-150., 2*pi/3, 2.}, 1e-6));
  EXPECT_TRUE(cmp(S6{}, std::tuple{30., 4*pi/3, 2.}));

  EXPECT_TRUE(cmp(S3{}, std::array{2., 1.1*pi, 0.6*pi}));
  EXPECT_TRUE(cmp(S3{}, std::array{2., 0.5*pi, 0.6*pi}));
  EXPECT_TRUE(cmp(S3{}, std::array{-2., 0.5*pi, 0.6*pi}));
  EXPECT_TRUE(cmp(S3{}, std::array{-2., 1.1*pi, 0.6*pi}));
  EXPECT_TRUE(cmp(S7{}, std::array{-240., pi/6, 2.}));
}
