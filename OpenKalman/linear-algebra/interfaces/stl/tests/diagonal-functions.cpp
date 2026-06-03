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
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/functions/copy_from.hpp"
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
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/functions/to_diagonal.hpp"
#include "linear-algebra/functions/make_constant_diagonal.hpp"
#include "linear-algebra/functions/make_identity_matrix.hpp"

TEST(stl_interfaces, to_diagonal)
{
  // Symmetrical diagonal matrices

  using A1 = double[1];
  A1 a1 {7};
  decltype(auto) d11 = to_diagonal(a1);
  using D11 = decltype(d11);
  static_assert(diagonal_matrix<D11>);
  static_assert(one_dimensional<D11>);
  static_assert(not dynamic_dimension<D11, 0>);
  static_assert(not dynamic_dimension<D11, 1>);
  static_assert(index_dimension_of_v<D11, 0> == 1);
  static_assert(index_dimension_of_v<D11, 1> == 1);
  EXPECT_EQ(constant_value(d11), 7);

  auto pa1 = attach_patterns(a1, std::tuple{Distance{}});
  decltype(auto) pd11 = to_diagonal(pa1);
  static_assert(compares_with_pattern_collection<decltype(pd11), std::tuple<Distance, Distance>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(constant_value(pd11), 7);

  using A11 = double[1][1];
  constexpr A11 a11 {{8}};
  decltype(auto) d111 = to_diagonal(a11);
  using D111 = decltype(d111);
  static_assert(diagonal_matrix<D111>);
  static_assert(one_dimensional<D111>);
  static_assert(not dynamic_dimension<D111, 0>);
  static_assert(not dynamic_dimension<D111, 1>);
  static_assert(index_dimension_of_v<D111, 0> == 1);
  static_assert(index_dimension_of_v<D111, 1> == 1);
  EXPECT_EQ(constant_value(d111), 8);

  auto pa11 = attach_patterns(a11, std::tuple{Distance{}, Angle<>{}});
  decltype(auto) pd111 = to_diagonal(pa11);
  static_assert(compares_with_pattern_collection<decltype(pd111), std::tuple<Distance, Distance, Angle<>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(constant_value(pd111), 8);

  using A3 = double[3];
  A3 a3 {1, 2, 3};
  decltype(auto) d33 = to_diagonal(a3);
  static_assert(diagonal_matrix<decltype(d33)>);
  static_assert(index_dimension_of_v<decltype(d33), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d33), 1> == 3);
  EXPECT_EQ((d33[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((d33[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((d33[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((d33[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((d33[std::array{1U, 1U}]), 2.);
  EXPECT_EQ((d33[std::array{1U, 2U}]), 0.);
  EXPECT_EQ((d33[std::array{2U, 0U}]), 0.);
  EXPECT_EQ((d33[std::array{2U, 1U}]), 0.);
  EXPECT_EQ((d33[std::array{2U, 2U}]), 3.);

  auto pa3 = attach_patterns(a3, std::tuple{Spherical{}});
  decltype(auto) pd33 = to_diagonal(pa3);
  static_assert(compares_with_pattern_collection<decltype(pd33), std::tuple<Spherical<>, Spherical<>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ((pd33[std::array{1U, 1U}]), 2.);
  EXPECT_EQ((pd33[std::array{2U, 1U}]), 0.);

  auto b3 = std::array {1., 2., 3.};
  auto mb3 = stdex::mdspan<double, stdex::extents<std::size_t, 3>>(b3.data());
  auto db33 = to_diagonal(mb3);
  static_assert(index_dimension_of_v<decltype(db33), 0> == 3);
  static_assert(index_dimension_of_v<decltype(db33), 1> == 3);
  EXPECT_EQ((db33[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((db33[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((db33[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((db33[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((db33[std::array{1U, 1U}]), 2.);
  EXPECT_EQ((db33[std::array{1U, 2U}]), 0.);
  EXPECT_EQ((db33[std::array{2U, 0U}]), 0.);
  EXPECT_EQ((db33[std::array{2U, 1U}]), 0.);
  EXPECT_EQ((db33[std::array{2U, 2U}]), 3.);

  auto pb3 = attach_patterns(a3, std::make_tuple(std::tuple{Distance{}, Polar<>{}}));
  decltype(auto) pdb33 = to_diagonal(pb3);
  static_assert(compares_with_pattern_collection<decltype(pdb33), std::tuple<std::tuple<Distance, Polar<>>, std::tuple<Distance, Polar<>>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ((pdb33[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((pdb33[std::array{2U, 2U}]), 3.);

  // Non-symmetrical diagonal matrices:

  decltype(auto) d2 = to_diagonal(a1, stdex::extents<std::size_t, 2>{});
  using D2 = decltype(d2);
  static_assert(diagonal_matrix<D2>);
  static_assert(not one_dimensional<D2>);
  static_assert(index_count_v<D2> == 1);
  static_assert(index_dimension_of_v<D2, 0> == 2);
  static_assert(index_dimension_of_v<D2, 1> == 1);
  static_assert(index_dimension_of_v<D2, 2> == 1);
  EXPECT_EQ((d2[0U]), 7.);
  EXPECT_EQ((d2[1U]), 0.);
  EXPECT_EQ(access(d2, std::array{0U, 0U, 0U}), 7.);
  EXPECT_EQ(access(d2, std::array{1U, 0U, 0U}), 0.);

  decltype(auto) pd2 = to_diagonal(a1, std::tuple{std::tuple{Distance{}, Angle<>{}}, Distance{}});
  static_assert(compares_with_pattern_collection<decltype(pd2), std::tuple<std::tuple<Distance, Angle<>>, Distance>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(access(pd2, std::array{0U, 0U}), 7.);
  EXPECT_EQ(access(pd2, std::array{1U, 0U}), 0.);

  decltype(auto) d21 = to_diagonal(a1, stdex::extents<std::size_t, 2, 1>{});
  using D21 = decltype(d21);
  static_assert(diagonal_matrix<D21>);
  static_assert(not one_dimensional<D21>);
  static_assert(index_count_v<D21> == 1);
  static_assert(index_dimension_of_v<D21, 0> == 2);
  static_assert(index_dimension_of_v<D21, 1> == 1);
  static_assert(index_dimension_of_v<D21, 2> == 1);
  EXPECT_EQ((d21[0U]), 7.);
  EXPECT_EQ((d21[1U]), 0.);
  EXPECT_EQ(access(d21, std::array{0U, 0U, 0U}), 7.);
  EXPECT_EQ(access(d21, std::array{1U, 0U, 0U}), 0.);

  decltype(auto) pd21 = to_diagonal(a1, std::tuple{Distance{}, std::tuple{Distance{}, Angle<>{}}});
  static_assert(compares_with_pattern_collection<decltype(pd21), std::tuple<Distance, std::tuple<Distance, Angle<>>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(access(pd21, std::array{0U, 0U}), 7.);
  EXPECT_EQ(access(pd21, std::array{1U, 0U}), 0.);

  auto d34 = to_diagonal(a3, stdex::extents<std::size_t, 3, 4>{});
  static_assert(diagonal_matrix<decltype(d34)>);
  static_assert(index_dimension_of_v<decltype(d34), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d34), 1> == 4);
  EXPECT_EQ((d34[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((d34[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((d34[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((d34[std::array{0U, 3U}]), 0.);
  EXPECT_EQ((d34[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((d34[std::array{1U, 1U}]), 2.);
  EXPECT_EQ((d34[std::array{1U, 2U}]), 0.);
  EXPECT_EQ((d34[std::array{1U, 3U}]), 0.);
  EXPECT_EQ((d34[std::array{2U, 0U}]), 0.);
  EXPECT_EQ((d34[std::array{2U, 1U}]), 0.);
  EXPECT_EQ((d34[std::array{2U, 2U}]), 3.);
  EXPECT_EQ((d34[std::array{2U, 3U}]), 0.);

  decltype(auto) pd34 = to_diagonal(a3, std::tuple{Spherical{}, std::tuple{Distance{}, Spherical{}}});
  static_assert(compares_with_pattern_collection<decltype(pd34), std::tuple<Spherical<>, std::tuple<Distance, Spherical<>>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(access(pd34, std::array{1U, 1U}), 2.);
  EXPECT_EQ(access(pd34, std::array{1U, 3U}), 0.);

  auto d43 = to_diagonal(a3, std::tuple<Dimensions<4>, Spherical<>>{});
  static_assert(diagonal_matrix<decltype(d43)>);
  static_assert(index_dimension_of_v<decltype(d43), 0> == 4);
  static_assert(index_dimension_of_v<decltype(d43), 1> == 3);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(d43)), Dimensions<4>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(d43)), Spherical<>>);
  EXPECT_EQ((d43[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((d43[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((d43[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((d43[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((d43[std::array{1U, 1U}]), 2.);
  EXPECT_EQ((d43[std::array{1U, 2U}]), 0.);
  EXPECT_EQ((d43[std::array{2U, 0U}]), 0.);
  EXPECT_EQ((d43[std::array{2U, 1U}]), 0.);
  EXPECT_EQ((d43[std::array{2U, 2U}]), 3.);
  EXPECT_EQ((d43[std::array{3U, 0U}]), 0.);
  EXPECT_EQ((d43[std::array{3U, 1U}]), 0.);
  EXPECT_EQ((d43[std::array{3U, 2U}]), 0.);

  decltype(auto) pd43 = to_diagonal(a3, std::tuple{std::tuple{Distance{}, Spherical<>{}}, Spherical<>{}});
  static_assert(compares_with_pattern_collection<decltype(pd43), std::tuple<std::tuple<Distance, Spherical<>>, Spherical<>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(access(pd43, std::array{2U, 2U}), 3.);
  EXPECT_EQ(access(pd43, std::array{3U, 0U}), 0.);

  // Zero

  using Z3 = F0[3];
  Z3 z3;
  decltype(auto) zd33 = to_diagonal(z3);
  using ZD33 = decltype(zd33);
  static_assert(diagonal_matrix<ZD33>);
  static_assert(zero<ZD33>);
  static_assert(index_dimension_of_v<decltype(zd33), 0> == 3);
  static_assert(index_dimension_of_v<decltype(zd33), 1> == 3);
  static_assert(index_dimension_of_v<ZD33, 0> == 3);
  static_assert(index_dimension_of_v<ZD33, 1> == 3);
  EXPECT_EQ(constant_value(zd33), 0);
  EXPECT_EQ((zd33[std::array{0U, 0U}]), 0.);
  EXPECT_EQ((zd33[std::array{1U, 1U}]), 0.);
  EXPECT_EQ((zd33[std::array{2U, 2U}]), 0.);

  auto c0_23 = make_zero<double>(std::tuple<Polar<>, Spherical<>>{});
  auto d0_23 = to_diagonal(c0_23);
  EXPECT_EQ(constant_value(d0_23), 0);
  static_assert(constant_object<decltype(d0_23)>);
  static_assert(zero<decltype(d0_23)>);
  static_assert(index_dimension_of_v<decltype(d0_23), 0> == 2);
  static_assert(index_dimension_of_v<decltype(d0_23), 1> == 2);
  static_assert(index_dimension_of_v<decltype(d0_23), 2> == 3);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(d0_23)), Polar<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(d0_23)), Polar<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<2>(d0_23)), Spherical<>>);

  auto c0_32 = make_zero<double>(std::tuple<Spherical<>, Dimensions<2>>{});
  auto d0_32 = to_diagonal(c0_32);
  EXPECT_EQ(constant_value(d0_32), 0);
  static_assert(constant_object<decltype(d0_32)>);
  static_assert(zero<decltype(d0_32)>);
  static_assert(index_dimension_of_v<decltype(d0_32), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d0_32), 1> == 3);
  static_assert(index_dimension_of_v<decltype(d0_32), 2> == 2);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(d0_32)), Spherical<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(d0_32)), Spherical<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<2>(d0_32)), Dimensions<2>>);

  // Constant diagonal

  auto c5d = make_constant(0.5, std::array {Any{Spherical{}}});
  static_assert(constant_object<decltype(c5d)>);
  auto dc5 = to_diagonal(c5d);
  EXPECT_EQ(constant_value(dc5), 0.5);
  static_assert(stdex::same_as<element_type_of_t<decltype(dc5)>, double>);
  static_assert(not constant_object<decltype(dc5)>);
  static_assert(constant_diagonal_object<decltype(dc5)>);
  EXPECT_EQ(get_index_extent(dc5, 0U), 3);
  EXPECT_EQ(get_index_extent(dc5, 1U), 3);
  EXPECT_TRUE(compare_pattern_collections(get_pattern_collection(dc5), std::tuple{Spherical{}, Spherical{}}));
  EXPECT_EQ((dc5[std::array{0U, 0U}]), 0.5);
  EXPECT_EQ((dc5[std::array{0U, 1U}]), 0.0);
  EXPECT_EQ((dc5[std::array{0U, 2U}]), 0.0);
  EXPECT_EQ((dc5[std::array{1U, 0U}]), 0.0);
  EXPECT_EQ((dc5[std::array{1U, 1U}]), 0.5);
  EXPECT_EQ((dc5[std::array{1U, 2U}]), 0.0);
  EXPECT_EQ((dc5[std::array{2U, 0U}]), 0.0);
  EXPECT_EQ((dc5[std::array{2U, 1U}]), 0.0);
  EXPECT_EQ((dc5[std::array{2U, 2U}]), 0.5);

  auto c33 = make_constant_diagonal(F5{}, stdex::extents<std::size_t, 3, 3>{});
  static_assert(stdex::same_as<element_type_of_t<decltype(c33)>, double>);
  static_assert(not constant_object<decltype(c33)>);
  static_assert(constant_diagonal_object<decltype(c33)>);
  static_assert(values::fixed_value_of_v<decltype(constant_value(c33))> == 5.0);
  EXPECT_EQ((c33[std::array{0U, 0U}]), 5.0);
  EXPECT_EQ((c33[std::array{0U, 1U}]), 0.0);
  EXPECT_EQ((c33[std::array{0U, 2U}]), 0.0);
  EXPECT_EQ((c33[std::array{1U, 0U}]), 0.0);
  EXPECT_EQ((c33[std::array{1U, 1U}]), 5.0);
  EXPECT_EQ((c33[std::array{1U, 2U}]), 0.0);
  EXPECT_EQ((c33[std::array{2U, 0U}]), 0.0);
  EXPECT_EQ((c33[std::array{2U, 1U}]), 0.0);
  EXPECT_EQ((c33[std::array{2U, 2U}]), 5.0);

  // Identity matrices

  auto i32 = make_identity_matrix<double>(std::array {Any{Spherical{}}, Any{Polar{}}});
  static_assert(identity_matrix<decltype(i32)>);
  EXPECT_EQ(constant_value(i32), 1);
  static_assert(stdex::same_as<element_type_of_t<decltype(i32)>, double>);
  static_assert(not constant_object<decltype(i32)>);
  EXPECT_EQ(get_index_extent(i32, 0U), 3);
  EXPECT_EQ(get_index_extent(i32, 1U), 2);
  EXPECT_TRUE(compare_pattern_collections(get_pattern_collection(i32), std::tuple{Spherical{}, Polar{}}));
  EXPECT_EQ((i32[std::array{0U, 0U}]), 1.0);
  EXPECT_EQ((i32[std::array{0U, 1U}]), 0.0);
  EXPECT_EQ((i32[std::array{1U, 0U}]), 0.0);
  EXPECT_EQ((i32[std::array{1U, 1U}]), 1.0);
  EXPECT_EQ((i32[std::array{2U, 0U}]), 0.0);
  EXPECT_EQ((i32[std::array{2U, 1U}]), 0.0);
}

#include "linear-algebra/concepts/vector.hpp"
#include "linear-algebra/functions/diagonal_of.hpp"

TEST(stl_interfaces, diagonal_of)
{
  using A1 = double[1];
  A1 a1 {7};
  decltype(auto) d1 = diagonal_of(a1);
  using D1 = decltype(d1);
  static_assert(diagonal_matrix<D1>);
  static_assert(one_dimensional<D1>);
  static_assert(not dynamic_dimension<D1, 0>);
  static_assert(not dynamic_dimension<D1, 1>);
  static_assert(index_dimension_of_v<D1, 0> == 1);
  static_assert(index_dimension_of_v<D1, 1> == 1);
  EXPECT_EQ(constant_value(d1), 7);

  auto pa11 = attach_patterns(a1, std::tuple{Distance{}, Distance{}});
  decltype(auto) pd1 = diagonal_of(pa11);
  static_assert(compares_with_pattern_collection<decltype(pd1), std::tuple<Distance>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ(constant_value(pd1), 7);

  using A33 = double[3][3];
  A33 a33 {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  static_assert(not one_dimensional<A33>);
  decltype(auto) d3 = diagonal_of(a33);
  static_assert(vector<decltype(d3)>);
  static_assert(index_dimension_of_v<decltype(d3), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d3), 1> == 1);
  EXPECT_EQ((d3[0U]), 1.);
  EXPECT_EQ((d3[1U]), 5.);
  EXPECT_EQ((d3[2U]), 9.);

  auto pa33 = attach_patterns(a33, std::tuple{Spherical{}, Spherical{}});
  decltype(auto) pd3 = diagonal_of(pa33);
  static_assert(compares_with_pattern_collection<decltype(pd3), std::tuple<Spherical<>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ((pd3[0U]), 1.);
  EXPECT_EQ((pd3[1U]), 5.);
  EXPECT_EQ((pd3[2U]), 9.);

  using A34 = double[3][4];
  A34 a34 {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  static_assert(not one_dimensional<A34>);
  decltype(auto) d34 = diagonal_of(a34);
  static_assert(vector<decltype(d34)>);
  static_assert(index_dimension_of_v<decltype(d34), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d34), 1> == 1);
  EXPECT_EQ((d34[0U]), 1.);
  EXPECT_EQ((d34[1U]), 6.);
  EXPECT_EQ((d34[2U]), 11.);

  decltype(auto) p34 = attach_patterns(a34, std::tuple{Spherical{}, std::tuple{Spherical{}, Distance{}}});
  decltype(auto) pd34 = diagonal_of(p34);
  static_assert(compares_with_pattern_collection<decltype(pd34), std::tuple<Spherical<>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ((pd34[0U]), 1.);
  EXPECT_EQ((pd34[1U]), 6.);
  EXPECT_EQ((pd34[2U]), 11.);

  using A43 = double[4][3];
  A43 a43 {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
  static_assert(not one_dimensional<A43>);
  decltype(auto) d43 = diagonal_of(a43);
  static_assert(vector<decltype(d43)>);
  static_assert(index_dimension_of_v<decltype(d43), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d43), 1> == 1);
  EXPECT_EQ((d43[0U]), 1.);
  EXPECT_EQ((d43[1U]), 5.);
  EXPECT_EQ((d43[2U]), 9.);

  decltype(auto) p43 = attach_patterns(a43, std::tuple{std::tuple{Spherical{}, Distance{}}, Spherical{}});
  decltype(auto) pd43 = diagonal_of(p43);
  static_assert(compares_with_pattern_collection<decltype(pd43), std::tuple<Spherical<>>, &stdex::is_eq, applicability::guaranteed>);
  EXPECT_EQ((pd43[0U]), 1.);
  EXPECT_EQ((pd43[1U]), 5.);
  EXPECT_EQ((pd43[2U]), 9.);

  using Z35 = F0[3][5];
  static_assert(diagonal_matrix<Z35>);
  Z35 z35;
  auto zd3 = diagonal_of(z35);
  using ZD3 = decltype(zd3);
  static_assert(diagonal_matrix<ZD3>);
  static_assert(zero<ZD3>);
  static_assert(index_dimension_of_v<ZD3, 0> == 3);
  static_assert(index_dimension_of_v<ZD3, 1> == 1);
  EXPECT_EQ(constant_value(zd3), 0);
  EXPECT_EQ(zd3[0U], 0.);
  EXPECT_EQ(zd3[1U], 0.);
  EXPECT_EQ(zd3[2U], 0.);

  auto c0_23 = make_zero<double>(std::tuple{Polar{}, std::tuple{Polar{}, Distance{}}});
  auto d0_23 = diagonal_of(c0_23);
  EXPECT_EQ(constant_value(d0_23), 0);
  static_assert(constant_object<decltype(d0_23)>);
  static_assert(zero<decltype(d0_23)>);
  static_assert(index_dimension_of_v<decltype(d0_23), 0> == 2);
  static_assert(index_dimension_of_v<decltype(d0_23), 1> == 1);
  static_assert(index_dimension_of_v<decltype(d0_23), 2> == 1);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(d0_23)), Polar<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(d0_23)), Dimensions<1>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<2>(d0_23)), Dimensions<1>>);

  auto c4_43 = make_constant_diagonal(F4{}, std::tuple{std::tuple{Spherical{}, Distance{}}, Spherical{}});
  auto d4_43 = diagonal_of(c4_43);
  EXPECT_EQ(constant_value(d4_43), 4);
  static_assert(constant_object<decltype(d4_43)>);
  static_assert(index_dimension_of_v<decltype(d4_43), 0> == 3);
  static_assert(index_dimension_of_v<decltype(d4_43), 1> == 1);
  static_assert(index_dimension_of_v<decltype(d4_43), 2> == 1);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(d4_43)), Spherical<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(d4_43)), Dimensions<1>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<2>(d4_43)), Dimensions<1>>);

  auto id_43 = make_identity_matrix<double>(std::tuple{std::tuple{Spherical{}, Distance{}}, Spherical{}});
  static_assert(identity_matrix<decltype(id_43)>);
  static_assert(constant_diagonal_object<decltype(id_43)>);
  auto di_43 = diagonal_of(id_43);
  EXPECT_EQ(constant_value(di_43), 1);
  static_assert(index_dimension_of_v<decltype(di_43), 0> == 3);
  static_assert(index_dimension_of_v<decltype(di_43), 1> == 1);
  static_assert(index_dimension_of_v<decltype(di_43), 2> == 1);
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(di_43)), Spherical<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(di_43)), Dimensions<1>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<2>(di_43)), Dimensions<1>>);

  using A23c = std::complex<double>[2][3];
  A23c a23c {{std::complex{1.,.1}, std::complex{2.,.2}, std::complex{3.,.3}},
            {std::complex{4.,-.4}, std::complex{5.,-.5}, std::complex{6.,-.6}}};
  std::complex<double> a2c[2] {std::complex{1.,.1}, std::complex{5.,-.5}};
  EXPECT_TRUE(is_near(diagonal_of(a23c), a2c));
}

