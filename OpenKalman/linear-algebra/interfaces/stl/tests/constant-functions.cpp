/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/functions/copy_from.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"

using namespace OpenKalman;

using namespace OpenKalman;
using namespace OpenKalman::test;
using namespace OpenKalman::patterns;

using stdex::numbers::pi;

namespace
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
  using N4 = std::integral_constant<std::size_t, 3>;

  using F0 = values::fixed_value<double, 0>;
  using F1 = values::fixed_value<double, 1>;
  using F2 = values::fixed_value<double, 2>;
  using F3 = values::fixed_value<double, 3>;
  using F4 = values::fixed_value<double, 4>;
  using F5 = values::fixed_value<double, 5>;
}

#include "linear-algebra/concepts/dynamic_dimension.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/functions/make_constant.hpp"

TEST(stl_interfaces, make_constant)
{
  auto c7 = make_constant(7.5, stdex::extents<std::size_t, 2, 3>{});
  EXPECT_EQ((c7[std::array{0U, 0U}]), 7.5);
  EXPECT_EQ((c7[std::array{0U, 1U}]), 7.5);
  EXPECT_EQ((c7[std::array{0U, 2U}]), 7.5);
  EXPECT_EQ((c7[std::array{1U, 0U}]), 7.5);
  EXPECT_EQ((c7[std::array{1U, 1U}]), 7.5);
  EXPECT_EQ((c7[std::array{1U, 2U}]), 7.5);
  static_assert(constant_object<decltype(c7)>);
  EXPECT_EQ(constant_value(c7), 7.5);

  auto c8 = make_constant(8.5, std::tuple<N2, N3, N4>{});
  EXPECT_EQ((c8[std::array{0U, 0U, 0U}]), 8.5);
  EXPECT_EQ((c8[std::array{0U, 1U, 1U}]), 8.5);
  EXPECT_EQ((c8[std::array{0U, 2U, 2U}]), 8.5);
  EXPECT_EQ((c8[std::array{1U, 0U, 3U}]), 8.5);
  EXPECT_EQ((c8[std::array{1U, 1U, 0U}]), 8.5);
  EXPECT_EQ((c8[std::array{1U, 2U, 1U}]), 8.5);
  static_assert(constant_object<decltype(c8)>);
  EXPECT_EQ(constant_value(c8), 8.5);

  using A23 = double[2][3];
  A23 a23 {{5, 5, 5}, {5, 5, 5}};

  auto m23 = make_constant(F5{}, std::tuple<Polar<>, Spherical<>>{});
  static_assert(not dynamic_dimension<decltype(m23), 0>);
  static_assert(not dynamic_dimension<decltype(m23), 1>);
  static_assert(index_dimension_of_v<decltype(m23), 0> == 2);
  static_assert(index_dimension_of_v<decltype(m23), 1> == 3);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(m23)), Polar<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(m23)), Spherical<>>);
  EXPECT_EQ(constant_value(m23), 5);
  EXPECT_EQ((m23[std::array{0U, 0U}]), 5);
  EXPECT_EQ((m23[std::array{0U, 1U}]), 5);
  EXPECT_EQ((m23[std::array{0U, 2U}]), 5);
  EXPECT_EQ((m23[std::array{1U, 0U}]), 5);
  EXPECT_EQ((m23[std::array{1U, 1U}]), 5);
  EXPECT_EQ((m23[std::array{1U, 2U}]), 5);
  static_assert(constant_object<decltype(m23)>);
  EXPECT_EQ(constant_value(m23), 5);
  static_assert(values::fixed_value_of_v<decltype(constant_value(m23))> == 5);

  auto m2x_3 = make_constant(F5{}, stdex::extents<std::size_t, 2, stdex::dynamic_extent>{3});
  static_assert(not dynamic_dimension<decltype(m2x_3), 0>);
  static_assert(dynamic_dimension<decltype(m2x_3), 1>);
  static_assert(index_dimension_of_v<decltype(m2x_3), 0> == 2);
  EXPECT_EQ(get_index_extent<1>(m2x_3), 3);
  EXPECT_EQ(constant_value(m2x_3), 5);
  EXPECT_TRUE(is_near(m2x_3, a23));

  auto mx3_2 = make_constant(F5{}, std::tuple{2U, N3{}});
  static_assert(dynamic_dimension<decltype(mx3_2), 0>);
  static_assert(not dynamic_dimension<decltype(mx3_2), 1>);
  static_assert(index_dimension_of_v<decltype(mx3_2), 1> == 3);
  EXPECT_EQ(get_index_extent<0>(mx3_2), 2);
  EXPECT_EQ(constant_value(mx3_2), 5);
  EXPECT_TRUE(is_near(mx3_2, a23));

  auto mxx_23 = make_constant(F5{}, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>{2, 3});
  static_assert(dynamic_dimension<decltype(mxx_23), 0>);
  static_assert(dynamic_dimension<decltype(mxx_23), 1>);
  EXPECT_EQ(get_index_extent<0>(mxx_23), 2);
  EXPECT_EQ(get_index_extent<1>(mxx_23), 3);
  EXPECT_EQ(constant_value(mxx_23), 5);
  EXPECT_TRUE(is_near(mxx_23, a23));

  auto m23d = make_constant(5_uz, stdex::extents<std::size_t, 2, 3>{});
  static_assert(not dynamic_dimension<decltype(m23d), 0>);
  static_assert(not dynamic_dimension<decltype(m23d), 1>);
  static_assert(index_dimension_of_v<decltype(m23d), 0> == 2);
  static_assert(index_dimension_of_v<decltype(m23d), 1> == 3);
  EXPECT_EQ(constant_value(m23d), 5);
  EXPECT_TRUE(is_near(m23d, a23));

  auto m2x_3d = make_constant(5_uz, stdex::extents<std::size_t, 2, stdex::dynamic_extent>{3});
  static_assert(not dynamic_dimension<decltype(m2x_3d), 0>);
  static_assert(dynamic_dimension<decltype(m2x_3d), 1>);
  static_assert(index_dimension_of_v<decltype(m2x_3d), 0> == 2);
  EXPECT_EQ(get_index_extent<1>(m2x_3d), 3);
  EXPECT_EQ(constant_value(m2x_3d), 5);
  EXPECT_TRUE(is_near(m2x_3d, a23));

  auto mx3_2d = make_constant(5_uz, stdex::extents<std::size_t, stdex::dynamic_extent, 3>{2});
  static_assert(dynamic_dimension<decltype(mx3_2d), 0>);
  static_assert(not dynamic_dimension<decltype(mx3_2d), 1>);
  static_assert(index_dimension_of_v<decltype(mx3_2d), 1> == 3);
  EXPECT_EQ(get_index_extent<0>(mx3_2d), 2);
  EXPECT_EQ(constant_value(mx3_2d), 5);
  EXPECT_TRUE(is_near(mx3_2d, a23));

  auto mxx_23d = make_constant(5_uz, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>{2, 3});
  static_assert(dynamic_dimension<decltype(mxx_23d), 0>);
  static_assert(dynamic_dimension<decltype(mxx_23d), 1>);
  EXPECT_EQ(get_index_extent<0>(mxx_23d), 2);
  EXPECT_EQ(get_index_extent<1>(mxx_23d), 3);
  EXPECT_EQ(constant_value(mxx_23d), 5);
  EXPECT_TRUE(is_near(mxx_23d, a23));
}

#include "linear-algebra/functions/make_zero.hpp"

TEST(stl_interfaces, make_zero)
{
  auto z23 = make_zero<double>(std::tuple<Polar<>, Spherical<>>{});
  EXPECT_EQ((z23[std::array{0U, 0U}]), 0);
  EXPECT_EQ((z23[std::array{0U, 1U}]), 0);
  EXPECT_EQ((z23[std::array{0U, 2U}]), 0);
  EXPECT_EQ((z23[std::array{1U, 0U}]), 0);
  EXPECT_EQ((z23[std::array{1U, 1U}]), 0);
  EXPECT_EQ((z23[std::array{1U, 2U}]), 0);
  static_assert(zero<decltype(z23)>);
  EXPECT_EQ(constant_value(z23), 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z23))> == 0);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(z23)), Polar<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(z23)), Spherical<>>);
}

