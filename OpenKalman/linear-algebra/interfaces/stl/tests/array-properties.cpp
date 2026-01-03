/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/stl/array-object.hpp"
#include "linear-algebra/tests/tests.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using stdex::numbers::pi;

namespace
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;

  using F0 = values::fixed_value<double, 0>;
  using F1 = values::fixed_value<double, 1>;
  using F2 = values::fixed_value<double, 2>;
  using F3 = values::fixed_value<double, 3>;
  using F4 = values::fixed_value<double, 4>;
  using F5 = values::fixed_value<double, 5>;

  using A1 = double[1];
  constexpr A1 a1 {7};
  using A11 = double[1][1];
  constexpr A11 a11 {{8}};
  using A111 = double[1][1][1];
  constexpr A111 a111 {{{9}}};

  using A3 = double[3];
  using A3c = const double[3];
  constexpr A3 a3 {1, 2, 3};
  constexpr A3c a3c {1, 2, 3};

  using A23 = double[2][3];
  using A23c = const double[2][3];
  constexpr A23 a23 {{1, 2, 3}, {4, 5, 6}};
  constexpr A23c a23c {{1, 2, 3}, {4, 5, 6}};

  using A234 = double[2][3][4];
  using A234c = const double[2][3][4];
  constexpr A234 a234 {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}};
  constexpr A234c a234c {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}};

  using Z1 = F0[1];
  using Z11 = F0[1][1];
  using Z111 = F0[1][1][1];
  constexpr Z1 z1 {F0{}};
  constexpr Z11 z11 {{F0{}}};
  constexpr Z111 z111 {{{F0{}}}};

  using Z3 = F0[3];
  constexpr Z3 z3 {F0{}, F0{}, F0{}};

  using Z23 = F0[2][3];
  constexpr Z23 z23 {{F0{}, F0{}, F0{}}, {F0{}, F0{}, F0{}}};

  using Z234 = F0[2][3][4];
  constexpr Z234 z234 {{{F0{}, F0{}, F0{}, F0{}},
                       {F0{}, F0{}, F0{}, F0{}},
                       {F0{}, F0{}, F0{}, F0{}}},
                      {{F0{}, F0{}, F0{}, F0{}},
                       {F0{}, F0{}, F0{}, F0{}},
                       {F0{}, F0{}, F0{}, F0{}}}};

  using I1 = F1[1];
  using I11 = F1[1][1];
  using I111 = F1[1][1][1];

}


#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/layout_of.hpp"

TEST(stl_interfaces, array_mdspan_derived_traits)
{
  static_assert(indexible<A3>);
  static_assert(indexible<A3c>);
  static_assert(indexible<A23>);
  static_assert(indexible<A23c>);
  static_assert(indexible<A234>);
  static_assert(indexible<A234c>);

  static_assert(count_indices(a3) == 1);
  static_assert(count_indices(a3c) == 1);
  static_assert(count_indices(a23) == 2);
  static_assert(count_indices(a23c) == 2);
  static_assert(count_indices(a234) == 3);
  static_assert(count_indices(a234c) == 3);

  static_assert(index_count_v<A3> == 1);
  static_assert(index_count_v<A3c> == 1);
  static_assert(index_count_v<A23> == 2);
  static_assert(index_count_v<A23c> == 2);
  static_assert(index_count_v<A234> == 3);
  static_assert(index_count_v<A234c> == 3);

  auto mdspan_a3 = get_mdspan(a3);
  auto mdspan_a23 = get_mdspan(a23);
  auto mdspan_a234 = get_mdspan(a234);
  EXPECT_EQ((mdspan_a3[std::array{1U}]), 2);
  EXPECT_EQ((mdspan_a23[std::array{1U, 2U}]), 6);
#ifdef __cpp_multidimensional_subscript
  EXPECT_EQ((mdspan_a23[0, 0]), 1);
  EXPECT_EQ((mdspan_a23[1, 1]), 5);
#else
  EXPECT_EQ((mdspan_a23(0, 0)), 1);
  EXPECT_EQ((mdspan_a23(1, 1)), 5);
#endif
  EXPECT_EQ((mdspan_a234[std::array{1U, 2U, 1U}]), 22);

  static_assert(stdex::same_as<element_type_of_t<A3>, double>);
  static_assert(stdex::same_as<element_type_of_t<A3c>, const double>);
  static_assert(stdex::same_as<element_type_of_t<A23>, double>);
  static_assert(stdex::same_as<element_type_of_t<A23c>, const double>);
  static_assert(stdex::same_as<element_type_of_t<A234>, double>);
  static_assert(stdex::same_as<element_type_of_t<A234c>, const double>);

  static_assert(stdex::same_as<layout_of_t<A3>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<A3c>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<A23>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<A23c>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<A234>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<A234c>, stdex::layout_right>);
}


#include "linear-algebra/traits/get_pattern_collection.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"
#include "linear-algebra/traits/get_index_extent.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"
#include "linear-algebra/traits/dynamic_index_count.hpp"
#include "linear-algebra/concepts/dynamic_dimension.hpp"
#include "linear-algebra/traits/tensor_order.hpp"
#include "linear-algebra/traits/max_tensor_order.hpp"
#include "linear-algebra/concepts/dimension_size_of_index_is.hpp"

TEST(stl_interfaces, array_extents_and_patterns)
{
  using namespace OpenKalman::patterns;

  static_assert(compare_pattern_collections(get_pattern_collection(a3), std::array{3U}));
  static_assert(compare_pattern_collections(get_pattern_collection(a3c), std::array{3U}));

  auto coll_a23 = get_pattern_collection(a23);
  auto coll_a23c = get_pattern_collection(a23c);
  static_assert(compare_pattern_collections(coll_a23, std::array{2U, 3U}));
  static_assert(compare_pattern_collections(coll_a23c, std::array{2U, 3U}));
  static_assert(euclidean_pattern_collection<decltype(coll_a23)>);
  static_assert(fixed_pattern_collection<decltype(coll_a23)>);
  static_assert(collections::index<decltype(coll_a23)>);
  static_assert(euclidean_pattern_collection<decltype(coll_a23c)>);
  static_assert(fixed_pattern_collection<decltype(coll_a23c)>);
  static_assert(collections::index<decltype(coll_a23c)>);
  static_assert(compare_pattern_collections(get_pattern_collection(a234), std::array{2U, 3U, 4U}));
  static_assert(compare_pattern_collections(get_pattern_collection(a234c), std::array{2U, 3U, 4U}));

  static_assert(compare(get_index_pattern(a23, N0{}), Dimensions<2>{}));
  static_assert(compare(get_index_pattern(a23, N1{}), Dimensions<3>{}));
  static_assert(compare(get_index_pattern(a23, N2{}), Dimensions<1>{}));
  static_assert(compare(get_index_pattern(a23, N3{}), Dimensions<1>{}));
  static_assert(compare(get_index_pattern(a23, 0U), 2U));
  static_assert(compare(get_index_pattern(a23, 1U), 3U));
  static_assert(compare(get_index_pattern(a23, 2U), 1U));
  static_assert(compare(get_index_pattern(a23, 3U), 1U));
  static_assert(compare(get_index_pattern(a23c, N0{}), Dimensions<2>{}));
  static_assert(compare(get_index_pattern(a23c, N1{}), Dimensions<3>{}));

  static_assert(get_index_extent(a23, N0{}) == 2);
  static_assert(get_index_extent(a23, N1{}) == 3);
  static_assert(get_index_extent(a23, N2{}) == 1);
  static_assert(get_index_extent(a23, N3{}) == 1);
  static_assert(get_index_extent(a23, 0U) == 2);
  static_assert(get_index_extent(a23, 1U) == 3);
  static_assert(get_index_extent(a23, 2U) == 1);
  static_assert(get_index_extent(a23, 3U) == 1);

  static_assert(index_dimension_of_v<A23, 0> == 2);
  static_assert(index_dimension_of_v<A23, 1> == 3);
  static_assert(index_dimension_of_v<A23, 2> == 1);
  static_assert(index_dimension_of_v<A23, 3> == 1);
  static_assert(index_dimension_of_v<A23c, 0> == 2);
  static_assert(index_dimension_of_v<A23c, 1> == 3);

  static_assert(dynamic_index_count_v<A3> == 0);
  static_assert(dynamic_index_count_v<A23> == 0);
  static_assert(dynamic_index_count_v<A234> == 0);

  static_assert(not dynamic_dimension<A3, 0>);
  static_assert(not dynamic_dimension<A23, 1>);
  static_assert(not dynamic_dimension<A234, 2>);

  static_assert(tensor_order(a1) == 0);
  static_assert(tensor_order(a111) == 0);
  static_assert(tensor_order(a3) == 1);
  static_assert(tensor_order(a23) == 2);
  static_assert(tensor_order(a234) == 3);

  static_assert(max_tensor_order_v<A1> == 0);
  static_assert(max_tensor_order_v<A111> == 0);
  static_assert(max_tensor_order_v<A3> == 1);
  static_assert(max_tensor_order_v<A23> == 2);
  static_assert(max_tensor_order_v<A234> == 3);
  static_assert(max_tensor_order_v<double[2][3][4][1][1]> == 3);

  static_assert(dimension_size_of_index_is<A23, 0, 2>);
  static_assert(dimension_size_of_index_is<A23, 0, 3, &stdex::is_neq>);
  static_assert(dimension_size_of_index_is<A23, 1, 3>);
  static_assert(dimension_size_of_index_is<A23, 1, 2, &stdex::is_neq>);
  static_assert(dimension_size_of_index_is<A23c, 0, 2>);
  static_assert(dimension_size_of_index_is<A23c, 0, 3, &stdex::is_neq>);
  static_assert(dimension_size_of_index_is<A23c, 1, 3>);
  static_assert(dimension_size_of_index_is<A23c, 1, 2, &stdex::is_neq>);
}


#include "linear-algebra/traits/patterns_match.hpp"
#include "linear-algebra/concepts/patterns_may_match_with.hpp"
#include "linear-algebra/concepts/patterns_match_with.hpp"
#include "linear-algebra/concepts/compares_with_pattern_collection.hpp"
#include "linear-algebra/traits/is_square_shaped.hpp"
#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/traits/is_one_dimensional.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/concepts/empty_object.hpp"
#include "linear-algebra/traits/is_vector.hpp"
#include "linear-algebra/concepts/vector.hpp"

TEST(stl_interfaces, array_shapes)
{
  using namespace OpenKalman::patterns;

  static_assert(patterns_match(a23, a23c));
  static_assert(patterns_match(a23, a23c, a23));

  static_assert(patterns_may_match_with<A23, A23c, A23, A23c>);
  static_assert(patterns_may_match_with<A23, A23c, double[2][3]>);
  static_assert(not patterns_may_match_with<A23, double[3][3]>);

  static_assert(patterns_match_with<A23, A23c, A23c>);
  static_assert(patterns_match_with<A23, A23c, double[2][3]>);
  static_assert(not patterns_match_with<A23, double[3][3]>);

  static_assert(compares_with_pattern_collection<double[2][3], std::tuple<Dimensions<2>, Dimensions<3>>>);
  static_assert(compares_with_pattern_collection<double[2][3], std::tuple<Dimensions<3>, Dimensions<4>, Dimensions<2>>, &stdex::is_lt>);
  static_assert(compares_with_pattern_collection<double[3][4], std::tuple<Dimensions<2>, Dimensions<3>>, &stdex::is_neq>);
  static_assert(not compares_with_pattern_collection<double[3][4], std::tuple<Dimensions<2>, Dimensions<3>, Dimensions<1>>, &stdex::is_neq>); // b/c the third dimension is equal
  static_assert(compares_with_pattern_collection<double[3][4], std::tuple<Dimensions<2>, Dimensions<3>, Dimensions<1>>, &stdex::is_gteq>);
  static_assert(not compares_with_pattern_collection<double[2][3], std::tuple<Dimensions<4>, Dimensions<3>>>);
  static_assert(compares_with_pattern_collection<double[2][3], std::tuple<std::size_t, Dimensions<3>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with_pattern_collection<double[2][3], std::vector<std::size_t>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with_pattern_collection<double[2][3], std::vector<std::size_t>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with_pattern_collection<double[2][3], std::tuple<std::size_t, Dimensions<3>>, &stdex::is_eq, applicability::guaranteed>);

  static_assert(not is_square_shaped(a3));
  static_assert(not is_square_shaped(a23));
  static_assert(not is_square_shaped(a234));
  static_assert(compare(*is_square_shaped(a1), Dimensions<1>{}));
  static_assert(compare(*is_square_shaped(a11), Dimensions<1>{}));
  static_assert(compare(*is_square_shaped(a111), Dimensions<1>{}));
  double a22[2][2] {{1, 2}, {3, 4}};
  static_assert(compare(*is_square_shaped(a22), Dimensions<2>{}));
  double a222[2][2][2] {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  static_assert(compare(*is_square_shaped(a222), Dimensions<2>{}));

  static_assert(not square_shaped<A3, values::unbounded_size, applicability::permitted>);
  static_assert(not square_shaped<A23, values::unbounded_size, applicability::permitted>);
  static_assert(not square_shaped<A234, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<double[1]>);
  static_assert(square_shaped<double[2][2]>);
  static_assert(square_shaped<double[2][2][2]>);
  static_assert(not square_shaped<double[2][1][2], values::unbounded_size, applicability::permitted>);

  static_assert(not is_one_dimensional(a3));
  static_assert(not is_one_dimensional(a23));
  static_assert(not is_one_dimensional(a234));
  static_assert(is_one_dimensional(a1));
  static_assert(is_one_dimensional(a11));
  static_assert(is_one_dimensional(a111));
  static_assert(not is_one_dimensional(a22));
  static_assert(not is_one_dimensional(a222));

  static_assert(not one_dimensional<A3, values::unbounded_size, applicability::permitted>);
  static_assert(not one_dimensional<A23, values::unbounded_size, applicability::permitted>);
  static_assert(not one_dimensional<A234, values::unbounded_size, applicability::permitted>);
  static_assert(one_dimensional<double[1]>);
  static_assert(one_dimensional<double[1][1]>);
  static_assert(one_dimensional<double[1][1][1]>);
  static_assert(not one_dimensional<double[1][2][1], values::unbounded_size, applicability::permitted>);

  // no tests for empty_object, given that a legacy c++ array cannot have zero dimension.

  static_assert(is_vector<0>(a3));
  static_assert(not is_vector<1>(a3));
  static_assert(not is_vector<0>(a23));
  static_assert(not is_vector<1>(a23));
  static_assert(not is_vector<0>(a234));
  static_assert(not is_vector<1>(a234));
  static_assert(not is_vector<2>(a234));
  static_assert(is_vector<0>(a1));
  static_assert(is_vector<1>(a1));
  static_assert(is_vector<2>(a1));
  static_assert(is_vector<0>(a11));
  static_assert(is_vector<1>(a11));
  static_assert(is_vector<2>(a11));
  static_assert(is_vector<0>(a111));
  static_assert(is_vector<1>(a111));
  static_assert(is_vector<2>(a111));
  double a31[3][1] {{1}, {2}, {3}};
  static_assert(is_vector<0>(a31));
  static_assert(not is_vector<1>(a31));
  double a13[1][3] {{1, 2, 3}};
  static_assert(not is_vector<0>(a13));
  static_assert(is_vector<1>(a13));
  double a121[1][2][1] {{{1}, {2}}};
  static_assert(not is_vector<0>(a121));
  static_assert(is_vector<1>(a121));
  static_assert(not is_vector<2>(a121));
  double a113[1][1][3] {{{1, 2, 3}}};
  static_assert(not is_vector<0>(a113));
  static_assert(not is_vector<1>(a113));
  static_assert(is_vector<2>(a113));

  static_assert(vector<A3, 0>);
  static_assert(not vector<A3, 1, applicability::permitted>);
  static_assert(not vector<A23, 0, applicability::permitted>);
  static_assert(not vector<A23, 1, applicability::permitted>);
  static_assert(not vector<A234, 0, applicability::permitted>);
  static_assert(not vector<A234, 1, applicability::permitted>);
  static_assert(not vector<A234, 2, applicability::permitted>);
  static_assert(vector<double[1], 0>);
  static_assert(vector<double[1], 1>);
  static_assert(vector<double[1][1], 0>);
  static_assert(vector<double[1][1], 1>);
  static_assert(vector<double[3][1], 0>);
  static_assert(not vector<double[3][1], 1, applicability::permitted>);
  static_assert(not vector<double[1][3], 0, applicability::permitted>);
  static_assert(vector<double[1][3], 1>);
  static_assert(vector<double[1][1][1], 0>);
  static_assert(vector<double[1][1][1], 1>);
  static_assert(vector<double[1][1][1], 2>);
  static_assert(vector<double[1][1][1], 3>);
  static_assert(vector<double[3][1][1], 0>);
  static_assert(not vector<double[3][1][1], 1, applicability::permitted>);
  static_assert(not vector<double[3][1][1], 2, applicability::permitted>);
  static_assert(not vector<double[1][3][1], 0, applicability::permitted>);
  static_assert(vector<double[1][3][1], 1>);
  static_assert(not vector<double[1][3][1], 2, applicability::permitted>);
  static_assert(not vector<double[1][1][3], 0, applicability::permitted>);
  static_assert(not vector<double[1][1][3], 1, applicability::permitted>);
  static_assert(vector<double[1][1][3], 2>);
  static_assert(not vector<double[2][1][3], 0, applicability::permitted>);
  static_assert(not vector<double[2][1][3], 1, applicability::permitted>);
  static_assert(not vector<double[2][1][3], 2, applicability::permitted>);
}

#include "linear-algebra/concepts/index_collection_for.hpp"
#include "linear-algebra/traits/access.hpp"
#include "linear-algebra/traits/access_at.hpp"

TEST(stl_interfaces, array_indices)
{
  static_assert(index_collection_for<std::array<std::size_t, 0>, A1>);
  static_assert(index_collection_for<std::array<std::size_t, 1>, A1>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, A1>);
  static_assert(index_collection_for<std::vector<std::size_t>, A1>);
  static_assert(index_collection_for<std::tuple<N0>, A1>);
  static_assert(index_collection_for<std::tuple<N0, N0>, A1>);
  static_assert(not index_collection_for<std::tuple<N1>, A1>);

  static_assert(index_collection_for<std::array<std::size_t, 0>, A11>);
  static_assert(index_collection_for<std::array<std::size_t, 1>, A11>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, A11>);
  static_assert(index_collection_for<std::array<std::size_t, 3>, A11>);
  static_assert(index_collection_for<std::tuple<N0, N0>, A11>);
  static_assert(index_collection_for<std::tuple<N0, N0, N0>, A11>);
  static_assert(not index_collection_for<std::tuple<N1, N0>, A11>);
  static_assert(not index_collection_for<std::tuple<N0, N1>, A11>);

  static_assert(index_collection_for<std::array<std::size_t, 0>, A111>);
  static_assert(index_collection_for<std::array<std::size_t, 1>, A111>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, A111>);
  static_assert(index_collection_for<std::array<std::size_t, 3>, A111>);
  static_assert(index_collection_for<std::array<std::size_t, 4>, A111>);
  static_assert(index_collection_for<std::tuple<N0, N0, N0>, A111>);
  static_assert(index_collection_for<std::tuple<N0, N0, N0, N0>, A111>);
  static_assert(not index_collection_for<std::tuple<N1, N0, N0>, A111>);
  static_assert(not index_collection_for<std::tuple<N0, N1, N0>, A111>);
  static_assert(not index_collection_for<std::tuple<N0, N0, N1>, A111>);
  static_assert(not index_collection_for<std::tuple<N0, N0, N0, N1>, A111>);

  static_assert(not index_collection_for<std::array<std::size_t, 1>, A23>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, A23>);
  static_assert(index_collection_for<std::array<std::size_t, 3>, A23>);
  static_assert(index_collection_for<std::tuple<N0, N0>, A23>);
  static_assert(index_collection_for<std::tuple<N1, N2>, A23>);
  static_assert(index_collection_for<std::tuple<N1, N2, N0>, A23>);
  static_assert(not index_collection_for<std::tuple<N2, N2>, A23>);
  static_assert(not index_collection_for<std::tuple<N1, N3>, A23>);

  static_assert(access(a1, std::array{0U,0U}) == 7);
  static_assert(access(a1, std::array{0U,0U,0U}) == 7);
  static_assert(access(a11, std::array{0U,0U,0U}) == 8);
  static_assert(access(a111, std::array{0U,0U,0U}) == 9);
  static_assert(access(a3, std::array{0U}) == 1);
  static_assert(access(a3, std::array{1U}) == 2);
  static_assert(access(a3, std::array{2U}) == 3);
  static_assert(access(a3, std::array{0U,0U}) == 1);
  static_assert(access(a3, std::array{1U,0U}) == 2);
  static_assert(access(a3, std::array{2U,0U}) == 3);
  static_assert(access(a23, std::array{0U,0U}) == 1);
  static_assert(access(a23, std::array{0U,1U}) == 2);
  static_assert(access(a23, std::array{0U,2U}) == 3);
  EXPECT_TRUE(access(a23, std::array{1U,0U}) == 4);
  EXPECT_TRUE(access(a23, std::array{1U,1U}) == 5);
  EXPECT_TRUE(access(a23, std::array{1U,2U}) == 6);
  static_assert(access(a23c, std::array{0U,0U}) == 1);
  static_assert(access(a23c, std::array{0U,1U}) == 2);
  static_assert(access(a23c, std::array{0U,2U}) == 3);
  EXPECT_TRUE(access(a23c, std::array{1U,0U}) == 4);
  EXPECT_TRUE(access(a23c, std::array{1U,1U}) == 5);
  EXPECT_TRUE(access(a23c, std::array{1U,2U}) == 6);

  static_assert(access(a1, std::tuple{N0{},N0{}}) == 7);
  static_assert(access(a1, std::tuple{N0{},N0{},N0{}}) == 7);
  static_assert(access(a11, std::tuple{N0{},N0{},N0{}}) == 8);
  static_assert(access(a111, std::tuple{N0{},N0{},N0{}}) == 9);
  static_assert(access(a3, std::tuple{N0{}}) == 1);
  static_assert(access(a3, std::tuple{N1{}}) == 2);
  static_assert(access(a3, std::tuple{N2{}}) == 3);
  static_assert(access(a3, std::tuple{N0{},N0{}}) == 1);
  static_assert(access(a3, std::tuple{N1{},N0{}}) == 2);
  static_assert(access(a3, std::tuple{N2{},N0{}}) == 3);
  static_assert(access(a23, std::tuple{N0{},N0{}}) == 1);
  static_assert(access(a23, std::tuple{N0{},N1{}}) == 2);
  static_assert(access(a23, std::tuple{N0{},N2{}}) == 3);
  EXPECT_EQ(access(a23, std::tuple{N1{},N0{}}), 4);
  EXPECT_EQ(access(a23, std::tuple{N1{},N1{}}), 5);
  EXPECT_EQ(access(a23, std::tuple{N1{},N2{}}), 6);

  EXPECT_EQ(access(a1, std::vector{0U,0U}), 7);
  EXPECT_EQ(access(a1, std::vector{0U,0U,0U}), 7);
  EXPECT_EQ(access(a11, std::vector{0U,0U,0U}), 8);
  EXPECT_EQ(access(a111, std::vector{0U,0U,0U}), 9);
  EXPECT_EQ(access(a3, std::vector{0U}), 1);
  EXPECT_EQ(access(a3, std::vector{1U}), 2);
  EXPECT_EQ(access(a3, std::vector{2U}), 3);
  EXPECT_EQ(access(a3, std::vector{0U,0U}), 1);
  EXPECT_EQ(access(a3, std::vector{1U,0U}), 2);
  EXPECT_EQ(access(a3, std::vector{2U,0U}), 3);
  EXPECT_EQ(access(a23, std::vector{0U,0U}), 1);
  EXPECT_EQ(access(a23, std::vector{0U,1U}), 2);
  EXPECT_EQ(access(a23, std::vector{0U,2U}), 3);
  EXPECT_EQ(access(a23, std::vector{1U,0U}), 4);
  EXPECT_EQ(access(a23, std::vector{1U,1U}), 5);
  EXPECT_EQ(access(a23, std::vector{1U,2U}), 6);

  static_assert(access(a1, 0U, 0U) == 7);
  static_assert(access(a1, 0U, 0U, 0U) == 7);
  static_assert(access(a11, 0U, 0U, 0U) == 8);
  static_assert(access(a111, 0U, 0U, 0U) == 9);
  static_assert(access(a3, 0U) == 1);
  static_assert(access(a3, 1U) == 2);
  static_assert(access(a3, 2U) == 3);
  static_assert(access(a3, 0U, 0U) == 1);
  static_assert(access(a3, 1U, 0U) == 2);
  static_assert(access(a3, 2U, 0U) == 3);
  static_assert(access(a23, 0U, 0U) == 1);
  static_assert(access(a23, 0U, 1U) == 2);
  static_assert(access(a23, 0U, 2U) == 3);
  EXPECT_EQ(access(a23, 1U, 0U), 4);
  EXPECT_EQ(access(a23, 1U, 1U), 5);
  EXPECT_EQ(access(a23, 1U, 2U), 6);

  static_assert(access_at(a1, 0U) == 7);
  static_assert(access_at(a1, 0U, 0U) == 7);
  EXPECT_EQ(access_at(a1, std::vector{0U, 0U}), 7);
  EXPECT_ANY_THROW(access_at(a1, 1U));
  EXPECT_ANY_THROW(access_at(a1, 0U, 1U));
  EXPECT_ANY_THROW(access_at(a1, std::vector{0U, 1U}));
  static_assert(access_at(a11, 0U, 0U) == 8);
  static_assert(access_at(a11, 0U, 0U, 0U) == 8);
  EXPECT_EQ(access_at(a11, std::vector{0U, 0U, 0U}), 8);
  EXPECT_ANY_THROW(access_at(a11, 0U, 1U));
  EXPECT_ANY_THROW(access_at(a11, 0U, 0U, 1U));
  EXPECT_ANY_THROW(access_at(a11, std::vector{0U, 0U, 1U}));
  static_assert(access_at(a111, 0U, 0U, 0U) == 9);
  static_assert(access_at(a111, 0U, 0U, 0U, 0U) == 9);
  EXPECT_EQ(access_at(a111, std::vector{0U, 0U, 0U, 0U, 0U}), 9);
  EXPECT_ANY_THROW(access_at(a111, 0U, 0U, 1U));
  EXPECT_ANY_THROW(access_at(a111, 0U, 0U, 0U, 1U));
  EXPECT_ANY_THROW(access_at(a111, std::vector{0U, 0U, 0U, 1U}));
  static_assert(access_at(a3, 0U) == 1);
  static_assert(access_at(a3, 1U) == 2);
  static_assert(access_at(a3, 2U) == 3);
  EXPECT_ANY_THROW(access_at(a3, 3U));
  static_assert(access_at(a3, 0U, 0U) == 1);
  static_assert(access_at(a3, 1U, 0U) == 2);
  static_assert(access_at(a3, 2U, 0U) == 3);
  EXPECT_ANY_THROW(access_at(a3, 2U, 1U));
  EXPECT_ANY_THROW(access_at(a3, 3U, 0U));
  static_assert(access_at(a23, 0U, 0U) == 1);
  static_assert(access_at(a23, 0U, 1U) == 2);
  static_assert(access_at(a23, 0U, 2U) == 3);
  EXPECT_EQ(access_at(a23, 1U, 0U), 4);
  EXPECT_EQ(access_at(a23, 1U, 1U), 5);
  EXPECT_EQ(access_at(a23, 1U, 2U), 6);
  EXPECT_ANY_THROW(access_at(a23, 2U, 2U));
  EXPECT_ANY_THROW(access_at(a23, 1U, 3U));
  EXPECT_ANY_THROW(access_at(a23, 1U, 2U, 1U));

  A23 a23w {{1, 2, 3}, {4, 5, 6}};
  constexpr A23c a23w2 {{11, 12, 13}, {14, 15, 16}};
  access(a23w, 0U, 0U) = 11;
  access(a23w, 0U, 1U) = 12;
  access(a23w, 0U, 2U) = 13;
  access(a23w, 1U, 0U) = 14;
  access(a23w, 1U, 1U) = 15;
  access(a23w, 1U, 2U) = 16;
  EXPECT_TRUE(is_near(a23w, a23w2));

  access_at(a23w, 0U, 0U) = 1;
  access_at(a23w, 0U, 1U) = 2;
  access_at(a23w, 0U, 2U) = 3;
  access_at(a23w, 1U, 0U) = 4;
  access_at(a23w, 1U, 1U) = 5;
  access_at(a23w, 1U, 2U) = 6;
  EXPECT_ANY_THROW(access_at(a23w, 2U, 2U) = 6);
  EXPECT_TRUE(is_near(a23w, a23));
}

#include "linear-algebra/concepts/zero.hpp"
#include "linear-algebra/traits/triangle_type_of.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
//#include "linear-algebra/traits/hermitian_adapter_type_of.hpp"

TEST(stl_interfaces, array_special_matrices)
{
  static_assert(zero<Z1>);
  static_assert(zero<Z11>);
  static_assert(zero<Z111>);
  static_assert(zero<Z3>);
  static_assert(zero<Z23>);
  static_assert(zero<Z234>);

  static_assert(triangle_type_of_v<Z1> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<Z11> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<Z111> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<Z3> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<Z23> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<Z234> == triangle_type::diagonal);

  static_assert(triangle_type_of_v<A1> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<A11> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<A111> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<A3> == triangle_type::none);
  static_assert(triangle_type_of_v<A23> == triangle_type::none);
  static_assert(triangle_type_of_v<A234> == triangle_type::none);

  static_assert(triangular_matrix<Z1, triangle_type::diagonal>);
  static_assert(triangular_matrix<Z1, triangle_type::upper>);
  static_assert(triangular_matrix<Z1, triangle_type::lower>);
  static_assert(triangular_matrix<Z11, triangle_type::diagonal>);
  static_assert(triangular_matrix<Z11, triangle_type::upper>);
  static_assert(triangular_matrix<Z11, triangle_type::lower>);
  static_assert(triangular_matrix<Z111, triangle_type::diagonal>);
  static_assert(triangular_matrix<Z111, triangle_type::upper>);
  static_assert(triangular_matrix<Z111, triangle_type::lower>);
  static_assert(triangular_matrix<Z3, triangle_type::diagonal>);
  static_assert(triangular_matrix<Z3, triangle_type::upper>);
  static_assert(triangular_matrix<Z3, triangle_type::lower>);
  static_assert(triangular_matrix<Z23, triangle_type::diagonal>);
  static_assert(triangular_matrix<Z23, triangle_type::upper>);
  static_assert(triangular_matrix<Z23, triangle_type::lower>);
  static_assert(triangular_matrix<Z234, triangle_type::diagonal>);
  static_assert(triangular_matrix<Z234, triangle_type::upper>);
  static_assert(triangular_matrix<Z234, triangle_type::lower>);

  static_assert(triangular_matrix<A1, triangle_type::diagonal>);
  static_assert(triangular_matrix<A1, triangle_type::upper>);
  static_assert(triangular_matrix<A1, triangle_type::lower>);
  static_assert(triangular_matrix<A11, triangle_type::diagonal>);
  static_assert(triangular_matrix<A11, triangle_type::upper>);
  static_assert(triangular_matrix<A11, triangle_type::lower>);
  static_assert(triangular_matrix<A111, triangle_type::diagonal>);
  static_assert(triangular_matrix<A111, triangle_type::upper>);
  static_assert(triangular_matrix<A111, triangle_type::lower>);
  static_assert(not triangular_matrix<A234>);
  static_assert(triangular_matrix<I1, triangle_type::diagonal>);
  static_assert(triangular_matrix<I11, triangle_type::diagonal>);
  static_assert(triangular_matrix<I111, triangle_type::diagonal>);

  static_assert(diagonal_matrix<Z1>);
  static_assert(diagonal_matrix<Z11>);
  static_assert(diagonal_matrix<Z111>);
  static_assert(diagonal_matrix<Z3>);
  static_assert(diagonal_matrix<Z23>);
  static_assert(diagonal_matrix<Z234>);
  static_assert(diagonal_matrix<A1>);
  static_assert(diagonal_matrix<A11>);
  static_assert(diagonal_matrix<A111>);
  static_assert(not diagonal_matrix<A234>);
  static_assert(diagonal_matrix<I1>);
  static_assert(diagonal_matrix<I11>);
  static_assert(diagonal_matrix<I111>);

  static_assert(hermitian_matrix<Z1>);
  static_assert(hermitian_matrix<Z11>);
  static_assert(hermitian_matrix<Z111>);
  static_assert(not hermitian_matrix<Z3>);
  static_assert(not hermitian_matrix<Z23>);
  static_assert(not hermitian_matrix<Z234>);
  static_assert(hermitian_matrix<A1>);
  static_assert(hermitian_matrix<A11>);
  static_assert(hermitian_matrix<A111>);
  static_assert(not hermitian_matrix<A234>);
  static_assert(hermitian_matrix<I1>);
  static_assert(hermitian_matrix<I11>);
  static_assert(hermitian_matrix<I111>);

}

#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/traits/constant_value_of.hpp"
#include "linear-algebra/concepts/identity_matrix.hpp"

TEST(stl_interfaces, array_constants)
{
  static_assert(not constant_object<int>);
  static_assert(constant_object<Z1>);
  static_assert(constant_object<Z11>);
  static_assert(constant_object<Z111>);
  static_assert(constant_diagonal_object<Z1>);
  static_assert(constant_diagonal_object<Z11>);
  static_assert(constant_diagonal_object<Z111>);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z1))> == 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z11))> == 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z111))> == 0);

  static_assert(constant_object<A1>);
  static_assert(constant_object<A11>);
  static_assert(constant_object<A111>);
  static_assert(constant_diagonal_object<A1>);
  static_assert(constant_diagonal_object<A11>);
  static_assert(constant_diagonal_object<A111>);
  static_assert(constant_value(a1) == 7);
  static_assert(constant_value(a11) == 8);
  static_assert(constant_value(a111) == 9);

  static_assert(constant_object<Z3>);
  static_assert(constant_object<Z23>);
  static_assert(constant_object<Z234>);
  static_assert(constant_diagonal_object<Z3>);
  static_assert(constant_diagonal_object<Z23>);
  static_assert(constant_diagonal_object<Z234>);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z3))> == 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z23))> == 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z234))> == 0);

  static_assert(identity_matrix<I1>);
  static_assert(identity_matrix<I11>);
  static_assert(identity_matrix<I111>);
  static_assert(not identity_matrix<Z11>);
  static_assert(not identity_matrix<A11>);
}
