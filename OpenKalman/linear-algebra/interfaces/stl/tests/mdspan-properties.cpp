/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

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

  auto a1 = std::array{7.};
  using M1 = stdex::mdspan<double, stdex::extents<std::size_t, 1>>;
  constexpr M1 m1 {a1.data()};
  using M11 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 1>>;
  constexpr M11 m11 {a1.data()};
  using M111 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 1, 1>>;
  constexpr M111 m111 {a1.data()};
  using Mx = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent>>;
  constexpr Mx mx_1 {a1.data(), 1};

  const auto a3c = std::array{1., 2., 3.};
  using M3c = stdex::mdspan<const double, stdex::extents<std::size_t, 3>>;
  constexpr M3c m3c {a3c.data()};
  auto a3 = a3c;
  using M3 = stdex::mdspan<double, stdex::extents<std::size_t, 3>>;
  constexpr M3 m3 {a3.data()};
  constexpr Mx mx_3 {a3.data(), 3};

  using M31 = stdex::mdspan<double, stdex::extents<std::size_t, 3, 1>>;
  constexpr M31 m31 {a3.data()};
  using M13 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 3>>;
  constexpr M13 m13 {a3.data()};
  using M131 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 3, 1>>;
  constexpr M131 m131 {a3.data()};
  using M113 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 1, 3>>;
  constexpr M113 m113 {a3.data()};


  const auto a23c = std::array{1., 2., 3., 4., 5., 6.};
  using M23c = stdex::mdspan<const double, stdex::extents<std::size_t, 2, 3>>;
  constexpr M23c m23c {a23c.data()};
  auto a23 = a23c;
  using M23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>;
  constexpr M23 m23 {a23.data()};
  using M23l = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>, stdex::layout_left>;
  constexpr M23l m23l {a23.data()};
  using Mxx = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>>;
  constexpr Mxx mxx_23 {a23.data(), 2, 3};
  using M2x = stdex::mdspan<double, stdex::extents<std::size_t, 2, stdex::dynamic_extent>>;
  constexpr M2x m2x_3 {a23.data(), 3};
  using Mx3 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>;
  constexpr Mx3 mx3_2 {a23.data(), 2};

  auto a234c = std::array{
    1., 2., 3., 4.,
    5., 6., 7., 8.,
    9., 10., 11., 12.,
    13., 14., 15., 16.,
    17., 18., 19., 20.,
    21., 22., 23., 24.};
  using M234c = stdex::mdspan<const double, stdex::extents<std::size_t, 2, 3, 4>>;
  constexpr M234c m234c {a234c.data()};
  using M234 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, 4>>;
  constexpr M234 m234 {a234c.data()};
  using M234l = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, 4>, stdex::layout_left>;
  constexpr M234l m234l {a234c.data()};

  auto a22 = std::array{1., 2., 3., 4.};
  using M22 = stdex::mdspan<const double, stdex::extents<std::size_t, 2, 2>>;
  constexpr M22 m22 {a22.data()};
  constexpr Mxx mxx_22 {a22.data(), 2, 2};
  constexpr M2x m2x_2 {a22.data(), 2};
  using Mx2 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>;
  constexpr Mx2 mx2_2 {a22.data(), 2};

  auto a222 = std::array{1., 2., 3., 4., 5., 6., 7., 8.};
  using M222 = stdex::mdspan<const double, stdex::extents<std::size_t, 2, 2, 2>>;
  constexpr M222 m222 {a222.data()};

  auto za1 = std::array<F0, 1>{};
  using Z1 = stdex::mdspan<F0, stdex::extents<std::size_t, 1>>;
  constexpr Z1 z1 {za1.data()};
  using Z11 = stdex::mdspan<F0, stdex::extents<std::size_t, 1, 1>>;
  constexpr Z11 z11 {za1.data()};
  using Z111 = stdex::mdspan<F0, stdex::extents<std::size_t, 1, 1, 1>>;
  constexpr Z111 z111 {za1.data()};

  auto za3 = std::array<F0, 3>{};
  using Z3 = stdex::mdspan<F0, stdex::extents<std::size_t, 3>>;
  constexpr Z3 z3 {za3.data()};
  auto za23 = std::array<F0, 6>{};
  using Z23 = stdex::mdspan<F0, stdex::extents<std::size_t, 2, 3>>;
  constexpr Z23 z23 {za23.data()};
  auto za234 = std::array<F0, 24>{};
  using Z234 = stdex::mdspan<F0, stdex::extents<std::size_t, 2, 3, 4>>;
  constexpr Z234 z234 {za234.data()};

  auto ia1 = std::array<F1, 1>{};
  using I1 = stdex::mdspan<F1, stdex::extents<std::size_t, 1>>;
  constexpr I1 i1 {ia1.data()};
  using I11 = stdex::mdspan<F1, stdex::extents<std::size_t, 1, 1>>;
  constexpr I11 i11 {ia1.data()};
  using I111 = stdex::mdspan<F1, stdex::extents<std::size_t, 1, 1, 1>>;
  constexpr I111 i111 {ia1.data()};

  auto a0 = std::array<double, 0>{};
  using M0 = stdex::mdspan<double, stdex::extents<std::size_t, 0>>;
  constexpr M0 m0 {a0.data()};
  using M01 = stdex::mdspan<double, stdex::extents<std::size_t, 0, 1>>;
  constexpr M01 m01 {a0.data()};
  using M011 = stdex::mdspan<double, stdex::extents<std::size_t, 0, 1, 1>>;
  constexpr M011 m011 {a0.data()};

  using M03 = stdex::mdspan<double, stdex::extents<std::size_t, 0, 3>>;
  constexpr M03 m03 {a0.data()};
  using M20 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 0>>;
  constexpr M20 m20 {a0.data()};

  using M034 = stdex::mdspan<double, stdex::extents<std::size_t, 0, 3, 6>>;
  constexpr M034 m034 {a0.data()};
  using M204 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 0, 6>>;
  constexpr M204 m204 {a0.data()};
  using M230 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, 0>>;
  constexpr M230 m230 {a0.data()};
  using M004 = stdex::mdspan<double, stdex::extents<std::size_t, 0, 0, 6>>;
  constexpr M004 m004 {a0.data()};

}


#include "linear-algebra/interfaces/interfaces-defined.hpp"

TEST(stl_interfaces, mdspan_interface_properties_defined)
{
  static_assert(not interface::get_constant_defined_for<M23>);
  static_assert(not interface::is_square_defined_for<M23>);
  static_assert(not interface::is_triangular_adapter_defined_for<M23>);
  static_assert(not interface::is_hermitian_defined_for<M23>);
  static_assert(not interface::hermitian_adapter_type_defined_for<M23>);
}

#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/layout_of.hpp"

TEST(stl_interfaces, mdspan_derived_traits)
{
  static_assert(indexible<M3>);
  static_assert(indexible<M3c>);
  static_assert(indexible<M23>);
  static_assert(indexible<M23c>);
  static_assert(indexible<M234>);

  static_assert(count_indices(m1) == 0);
  static_assert(count_indices(m3c) == 1);
  static_assert(count_indices(m3) == 1);
  static_assert(count_indices(mx_3) == 1);
  static_assert(count_indices(m23c) == 2);
  static_assert(count_indices(m23) == 2);
  static_assert(count_indices(mxx_23) == 2);
  static_assert(count_indices(m2x_3) == 2);
  static_assert(count_indices(mx3_2) == 2);
  static_assert(count_indices(m23l) == 2);
  static_assert(count_indices(m234c) == 3);
  static_assert(count_indices(m234) == 3);
  static_assert(count_indices(m234l) == 3);
  static_assert(count_indices(z1) == 0);
  static_assert(count_indices(z11) == 0);
  static_assert(count_indices(z111) == 0);
  static_assert(count_indices(i1) == 0);
  static_assert(count_indices(i11) == 0);
  static_assert(count_indices(i111) == 0);
  static_assert(count_indices(m0) == 1);
  static_assert(count_indices(m01) == 1);
  static_assert(count_indices(m011) == 1);
  static_assert(count_indices(m03) == 2);
  static_assert(count_indices(m20) == 2);
  static_assert(count_indices(m034) == 3);
  static_assert(count_indices(m204) == 3);
  static_assert(count_indices(m230) == 3);
  static_assert(count_indices(m004) == 3);

  static_assert(index_count_v<M1> == 0);
  static_assert(index_count_v<M3c> == 1);
  static_assert(index_count_v<M3> == 1);
  static_assert(index_count_v<M23c> == 2);
  static_assert(index_count_v<M23> == 2);
  static_assert(index_count_v<Mxx> == 2);
  static_assert(index_count_v<M2x> == 2);
  static_assert(index_count_v<Mx3> == 2);
  static_assert(index_count_v<M23> == 2);
  static_assert(index_count_v<M23l> == 2);
  static_assert(index_count_v<M234c> == 3);
  static_assert(index_count_v<M234> == 3);
  static_assert(index_count_v<M234l> == 3);
  static_assert(index_count_v<M011> == 1);
  static_assert(index_count_v<M204> == 3);
  static_assert(index_count_v<M230> == 3);
  static_assert(index_count_v<M0> == 1);
  static_assert(index_count_v<M03> == 2);

  EXPECT_EQ((get_mdspan(m3)[std::array{1U}]), 2);
  EXPECT_EQ((m1[std::array{0U}]), 7);
  EXPECT_EQ((mx_1[std::array{0U}]), 7);
  EXPECT_EQ((m11[std::array{0U, 0U}]), 7);
  EXPECT_EQ((m111[std::array{0U, 0U, 0U}]), 7);
  EXPECT_EQ((m3[std::array{1U}]), 2);
  EXPECT_EQ((mx_3[std::array{2U}]), 3);
  EXPECT_EQ((m23[std::array{1U, 2U}]), 6);
  EXPECT_EQ((mxx_23[std::array{1U, 2U}]), 6);
  EXPECT_EQ((m2x_3[std::array{1U, 2U}]), 6);
  EXPECT_EQ((mx3_2[std::array{1U, 2U}]), 6);
#ifdef __cpp_multidimensional_subscript
  EXPECT_EQ((m23[0, 0]), 1);
  EXPECT_EQ((m23[1, 1]), 5);
  EXPECT_EQ((mxx_23[0, 0]), 1);
  EXPECT_EQ((mxx_23[1, 1]), 5);
#else
  EXPECT_EQ((m23(0, 0)), 1);
  EXPECT_EQ((m23(1, 1)), 5);
  EXPECT_EQ((mxx_23(0, 0)), 1);
  EXPECT_EQ((mxx_23(1, 1)), 5);
#endif
  EXPECT_EQ((m234[std::array{1U, 2U, 1U}]), 22);
  EXPECT_EQ((m234[std::array{0U, 1U, 2U}]), 7);
  EXPECT_EQ((m234l[std::array{1U, 2U, 1U}]), 12);
  EXPECT_EQ((m234l[std::array{0U, 1U, 2U}]), 15);
  EXPECT_EQ((z1[std::array{0U}]), 0);
  EXPECT_EQ((z11[std::array{0U, 0U}]), 0);
  EXPECT_EQ((z111[std::array{0U, 0U, 0U}]), 0);
  EXPECT_EQ((z3[std::array{1U}]), 0);
  EXPECT_EQ((z23[std::array{0U, 2U}]), 0);
  EXPECT_EQ((z234[std::array{1U, 0U, 2U}]), 0);
  EXPECT_EQ((i1[std::array{0U}]), 1);
  EXPECT_EQ((i11[std::array{0U, 0U}]), 1);
  EXPECT_EQ((i111[std::array{0U, 0U, 0U}]), 1);

  static_assert(stdex::same_as<element_type_of_t<M3>, double>);
  static_assert(stdex::same_as<element_type_of_t<M3c>, const double>);
  static_assert(stdex::same_as<element_type_of_t<M23>, double>);
  static_assert(stdex::same_as<element_type_of_t<Mxx>, double>);
  static_assert(stdex::same_as<element_type_of_t<M2x>, double>);
  static_assert(stdex::same_as<element_type_of_t<Mx3>, double>);
  static_assert(stdex::same_as<element_type_of_t<M23c>, const double>);
  static_assert(stdex::same_as<element_type_of_t<M23l>, double>);
  static_assert(stdex::same_as<element_type_of_t<M234>, double>);
  static_assert(stdex::same_as<element_type_of_t<M234c>, const double>);
  static_assert(stdex::same_as<element_type_of_t<M0>, double>);

  static_assert(internal::layout_mapping_policy<stdex::layout_left>);
  static_assert(internal::layout_mapping_policy<stdex::layout_right>);
  static_assert(internal::layout_mapping_policy<stdex::layout_stride>);

  static_assert(stdex::same_as<layout_of_t<M3>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M3c>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M23>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<Mxx>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M2x>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<Mx3>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M23c>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M23l>, stdex::layout_left>);
  static_assert(stdex::same_as<layout_of_t<M234>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M234c>, stdex::layout_right>);
  static_assert(stdex::same_as<layout_of_t<M0>, stdex::layout_right>);
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

TEST(stl_interfaces, mdspan_extents_and_patterns)
{
  using namespace OpenKalman::patterns;

  static_assert(compare_pattern_collections(get_pattern_collection(m3c), std::array{3U}));
  static_assert(compare_pattern_collections(get_pattern_collection(m3), std::array{3U}));

  auto coll_m23c = get_pattern_collection(m23c);
  auto coll_m23 = get_pattern_collection(m23);
  auto coll_mxx_23 = get_pattern_collection(mxx_23);
  auto coll_m2x_3 = get_pattern_collection(m2x_3);
  auto coll_mx3_2 = get_pattern_collection(mx3_2);
  auto coll_m23l = get_pattern_collection(m23l);
  static_assert(compare_pattern_collections(coll_m23c, std::array{2U, 3U}));
  static_assert(compare_pattern_collections(coll_m23, std::array{2U, 3U}));
  EXPECT_TRUE(compare_pattern_collections(coll_mxx_23, std::array{2U, 3U}));
  EXPECT_TRUE(compare_pattern_collections(coll_m2x_3, std::array{2U, 3U}));
  EXPECT_TRUE(compare_pattern_collections(coll_mx3_2, std::array{2U, 3U}));
  static_assert(compare_pattern_collections(coll_m23l, std::array{2U, 3U}));
  static_assert(euclidean_pattern_collection<decltype(coll_m23c)>);
  static_assert(euclidean_pattern_collection<decltype(coll_m23)>);
  static_assert(euclidean_pattern_collection<decltype(coll_mxx_23)>);
  static_assert(euclidean_pattern_collection<decltype(coll_m2x_3)>);
  static_assert(euclidean_pattern_collection<decltype(coll_mx3_2)>);
  static_assert(euclidean_pattern_collection<decltype(coll_m23l)>);
  static_assert(fixed_pattern_collection<decltype(coll_m23c)>);
  static_assert(fixed_pattern_collection<decltype(coll_m23)>);
  static_assert(not fixed_pattern_collection<decltype(coll_mxx_23)>);
  static_assert(not fixed_pattern_collection<decltype(coll_m2x_3)>);
  static_assert(not fixed_pattern_collection<decltype(coll_mx3_2)>);
  static_assert(fixed_pattern_collection<decltype(coll_m23l)>);
  static_assert(collections::index<decltype(coll_m23c)>);
  static_assert(collections::index<decltype(coll_m23)>);
  static_assert(collections::index<decltype(coll_mxx_23)>);
  static_assert(collections::index<decltype(coll_m2x_3)>);
  static_assert(collections::index<decltype(coll_mx3_2)>);
  static_assert(collections::index<decltype(coll_m23l)>);
  static_assert(compare_pattern_collections(get_pattern_collection(m234c), std::array{2U, 3U, 4U}));
  static_assert(compare_pattern_collections(get_pattern_collection(m234), std::array{2U, 3U, 4U}));
  static_assert(compare_pattern_collections(get_pattern_collection(m234l), std::array{2U, 3U, 4U}));

  static_assert(compare(get_index_pattern(m23, N0{}), Dimensions<2>{}));
  static_assert(compare(get_index_pattern(m23, N1{}), Dimensions<3>{}));
  static_assert(compare(get_index_pattern(m23, N2{}), Dimensions<1>{}));
  static_assert(compare(get_index_pattern(m23, N3{}), Dimensions<1>{}));
  static_assert(compare(get_index_pattern(m23, 0U), 2U));
  static_assert(compare(get_index_pattern(m23, 1U), 3U));
  static_assert(compare(get_index_pattern(m23, 2U), 1U));
  static_assert(compare(get_index_pattern(m23, 3U), 1U));
  static_assert(compare(get_index_pattern(mxx_23, 0U), 2U));
  static_assert(compare(get_index_pattern(mxx_23, 1U), 3U));
  static_assert(compare(get_index_pattern(mxx_23, 2U), 1U));
  static_assert(compare(get_index_pattern(mxx_23, 3U), 1U));
  static_assert(compare(get_index_pattern(m23c, N0{}), Dimensions<2>{}));
  static_assert(compare(get_index_pattern(m23c, N1{}), Dimensions<3>{}));
  static_assert(compare(get_index_pattern(m23l, N0{}), Dimensions<2>{}));
  static_assert(compare(get_index_pattern(m23l, N1{}), Dimensions<3>{}));

  static_assert(get_index_extent(m23, N0{}) == 2);
  static_assert(get_index_extent(m23, N1{}) == 3);
  static_assert(get_index_extent(m23, N2{}) == 1);
  static_assert(get_index_extent(m23, N3{}) == 1);
  static_assert(get_index_extent(m23, 0U) == 2);
  static_assert(get_index_extent(m23, 1U) == 3);
  static_assert(get_index_extent(m23, 2U) == 1);
  static_assert(get_index_extent(m23, 3U) == 1);
  static_assert(get_index_extent(mxx_23, 0U) == 2);
  static_assert(get_index_extent(mxx_23, 1U) == 3);
  static_assert(get_index_extent(mxx_23, 2U) == 1);
  static_assert(get_index_extent(mxx_23, 3U) == 1);
  static_assert(get_index_extent(m23l, 0U) == 2);
  static_assert(get_index_extent(m23l, 1U) == 3);
  static_assert(get_index_extent(m23l, 2U) == 1);
  static_assert(get_index_extent(m23l, 3U) == 1);

  static_assert(index_dimension_of_v<M23, 0> == 2);
  static_assert(index_dimension_of_v<M23, 1> == 3);
  static_assert(index_dimension_of_v<M23, 2> == 1);
  static_assert(index_dimension_of_v<M23, 3> == 1);
  static_assert(index_dimension_of_v<Mxx, 0> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Mxx, 1> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Mxx, 2> == 1);
  static_assert(index_dimension_of_v<Mxx, 3> == 1);
  static_assert(index_dimension_of_v<M2x, 0> == 2);
  static_assert(index_dimension_of_v<M2x, 1> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<M2x, 2> == 1);
  static_assert(index_dimension_of_v<M2x, 3> == 1);
  static_assert(index_dimension_of_v<Mx3, 0> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Mx3, 1> == 3);
  static_assert(index_dimension_of_v<Mx3, 2> == 1);
  static_assert(index_dimension_of_v<Mx3, 3> == 1);
  static_assert(index_dimension_of_v<M23c, 0> == 2);
  static_assert(index_dimension_of_v<M23c, 1> == 3);
  static_assert(index_dimension_of_v<M23l, 0> == 2);
  static_assert(index_dimension_of_v<M23l, 1> == 3);
  static_assert(index_dimension_of_v<M23l, 2> == 1);
  static_assert(index_dimension_of_v<M23l, 3> == 1);

  static_assert(dynamic_index_count_v<M3> == 0);
  static_assert(dynamic_index_count_v<M23> == 0);
  static_assert(dynamic_index_count_v<Mxx> == 2);
  static_assert(dynamic_index_count_v<M2x> == 1);
  static_assert(dynamic_index_count_v<Mx3> == 1);
  static_assert(dynamic_index_count_v<M23l> == 0);
  static_assert(dynamic_index_count_v<M234> == 0);

  static_assert(not dynamic_dimension<M3, 0>);
  static_assert(not dynamic_dimension<M23l, 1>);
  static_assert(not dynamic_dimension<M234, 2>);

  static_assert(tensor_order(m1) == 0);
  static_assert(tensor_order(m111) == 0);
  static_assert(tensor_order(m3) == 1);
  static_assert(tensor_order(m23) == 2);
  static_assert(tensor_order(mxx_23) == 2);
  static_assert(tensor_order(m2x_3) == 2);
  static_assert(tensor_order(mx3_2) == 2);
  static_assert(tensor_order(m23l) == 2);
  static_assert(tensor_order(m234) == 3);

  static_assert(max_tensor_order_v<M1> == 0);
  static_assert(max_tensor_order_v<M111> == 0);
  static_assert(max_tensor_order_v<M3> == 1);
  static_assert(max_tensor_order_v<M23> == 2);
  static_assert(max_tensor_order_v<Mxx> == 2);
  static_assert(max_tensor_order_v<M2x> == 2);
  static_assert(max_tensor_order_v<Mx3> == 2);
  static_assert(max_tensor_order_v<M23l> == 2);
  static_assert(max_tensor_order_v<M234> == 3);
  static_assert(max_tensor_order_v<stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, 4, 1, 1>>> == 3);
  static_assert(max_tensor_order_v<stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, 1, 1, 4>>> == 3);
  static_assert(max_tensor_order_v<stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, 0, 1, 4>>> == 0);
  static_assert(max_tensor_order_v<stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, stdex::dynamic_extent, 1, 4>>> == 4);
  static_assert(max_tensor_order_v<stdex::mdspan<double, stdex::extents<std::size_t, 2, 3, stdex::dynamic_extent, 0, 4>>> == 0);

  static_assert(dimension_size_of_index_is<M23, 0, 2>);
  static_assert(dimension_size_of_index_is<M23, 0, 3, &stdex::is_neq>);
  static_assert(dimension_size_of_index_is<M23, 1, 3>);
  static_assert(dimension_size_of_index_is<M23, 1, 2, &stdex::is_neq>);

  static_assert(not dimension_size_of_index_is<Mxx, 0, 2>);
  static_assert(not dimension_size_of_index_is<Mxx, 1, 3>);
  static_assert(dimension_size_of_index_is<Mxx, 0, 5, &stdex::is_eq, applicability::permitted>);
  static_assert(dimension_size_of_index_is<Mxx, 1, 6, &stdex::is_eq, applicability::permitted>);
  static_assert(not dimension_size_of_index_is<M2x, 0, 5>);
  static_assert(dimension_size_of_index_is<M2x, 0, 2>);
  static_assert(dimension_size_of_index_is<M2x, 1, 6, &stdex::is_eq, applicability::permitted>);
  static_assert(dimension_size_of_index_is<Mx3, 0, 5, &stdex::is_eq, applicability::permitted>);
  static_assert(dimension_size_of_index_is<Mx3, 1, 3>);
  static_assert(not dimension_size_of_index_is<Mx3, 1, 6>);

  static_assert(dimension_size_of_index_is<M23c, 0, 2>);
  static_assert(dimension_size_of_index_is<M23c, 0, 3, &stdex::is_neq>);
  static_assert(dimension_size_of_index_is<M23c, 1, 3>);
  static_assert(dimension_size_of_index_is<M23c, 1, 2, &stdex::is_neq>);
}


#include "linear-algebra/traits/patterns_match.hpp"
#include "linear-algebra/concepts/patterns_may_match_with.hpp"
#include "linear-algebra/concepts/patterns_match_with.hpp"
#include "linear-algebra/concepts/compares_with_pattern_collection.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/traits/is_square_shaped.hpp"
#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/traits/is_one_dimensional.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/concepts/empty_object.hpp"
#include "linear-algebra/traits/is_vector.hpp"
#include "linear-algebra/concepts/vector.hpp"

TEST(stl_interfaces, mdspan_shapes)
{
  using namespace OpenKalman::patterns;

  static_assert(patterns_match(m23, m23c));
  static_assert(patterns_match(m23, m23c, m23l));
  static_assert(patterns_match(m23, m23c, mxx_23));

  static_assert(patterns_may_match_with<M23, M23c, M23l, M23c>);
  static_assert(patterns_may_match_with<M23, M23c, M23l, Mxx, M2x, Mx3>);
  static_assert(not patterns_may_match_with<M23, stdex::mdspan<std::array<double, 9>, stdex::extents<std::size_t, 3, 3>>>);

  static_assert(patterns_match_with<M23, M23c, M23l>);
  static_assert(not patterns_match_with<M23, M23c, M23l, M2x>);
  static_assert(not patterns_match_with<M23, stdex::mdspan<std::array<double, 9>, stdex::extents<std::size_t, 3, 3>>>);

  static_assert(compares_with_pattern_collection<M23, std::tuple<Dimensions<2>, Dimensions<3>>>);
  static_assert(compares_with_pattern_collection<M23, std::tuple<Dimensions<3>, Dimensions<4>, Dimensions<2>>, &stdex::is_lt>);
  static_assert(compares_with_pattern_collection<M23, std::tuple<Dimensions<1>, Dimensions<2>>, &stdex::is_neq>);
  static_assert(not compares_with_pattern_collection<M23, std::tuple<Dimensions<1>, Dimensions<2>, Dimensions<1>>, &stdex::is_neq>); // b/c the third dimension is equal
  static_assert(compares_with_pattern_collection<M23, std::tuple<Dimensions<3>, Dimensions<4>, Dimensions<1>>, &stdex::is_lteq>);
  static_assert(not compares_with_pattern_collection<M23, std::tuple<Dimensions<4>, Dimensions<3>>>);
  static_assert(compares_with_pattern_collection<M23, std::tuple<std::size_t, Dimensions<3>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with_pattern_collection<M23, std::vector<std::size_t>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with_pattern_collection<M23, std::vector<std::size_t>, &stdex::is_lt, applicability::permitted>);
  static_assert(not compares_with_pattern_collection<M23, std::tuple<std::size_t, Dimensions<3>>, &stdex::is_eq, applicability::guaranteed>);
  static_assert(compares_with_pattern_collection<Mxx, std::tuple<std::size_t, Dimensions<3>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with_pattern_collection<Mx3, std::tuple<std::size_t, Dimensions<3>>, &stdex::is_eq, applicability::permitted>);
  static_assert(compares_with_pattern_collection<M2x, std::tuple<Dimensions<2>, Dimensions<3>>, &stdex::is_eq, applicability::permitted>);

  static_assert(compares_with_pattern_collection<M03, std::tuple<Dimensions<0>, Dimensions<3>>>);
  static_assert(compares_with_pattern_collection<M03, std::tuple<Dimensions<0>, Dimensions<3>, Dimensions<1>>>);

  static_assert(pattern_collection_for<std::tuple<Dimensions<2>, Dimensions<3>>, M23, applicability::guaranteed>);
  static_assert(pattern_collection_for<std::tuple<Polar<>, Spherical<>>, M23, applicability::guaranteed>);
  static_assert(pattern_collection_for<std::tuple<Dimensions<2>, Dimensions<3>, Dimensions<1>>, M23, applicability::guaranteed>);
  static_assert(pattern_collection_for<std::tuple<Polar<>, Spherical<>, Dimensions<1>>, M23, applicability::guaranteed>);
  static_assert(pattern_collection_for<std::tuple<Polar<>, Spherical<>, Angle<>>, M23, applicability::guaranteed>);
  static_assert(not pattern_collection_for<std::tuple<Dimensions<3>, Dimensions<4>, Dimensions<2>>, M23>);
  static_assert(not pattern_collection_for<std::tuple<Dimensions<1>, Dimensions<2>>, M23>);
  static_assert(pattern_collection_for<std::tuple<std::size_t, std::size_t>, M23>);
  static_assert(pattern_collection_for<std::array<std::size_t, 2>, M23>);
  static_assert(pattern_collection_for<std::tuple<std::size_t, std::size_t>, Mxx>);
  static_assert(pattern_collection_for<std::tuple<std::size_t, std::size_t, Dimensions<1>>, Mxx>);
  static_assert(pattern_collection_for<std::array<std::size_t, 2>, Mxx>);
  static_assert(pattern_collection_for<std::array<std::size_t, 3>, Mxx>);
  static_assert(pattern_collection_for<std::tuple<std::size_t, Dimensions<3>>, Mx3>);
  static_assert(pattern_collection_for<std::tuple<std::size_t, Dimensions<3>, Dimensions<1>>, Mx3>);
  static_assert(pattern_collection_for<std::tuple<Dimensions<2>, std::size_t>, M2x>);
  static_assert(pattern_collection_for<std::tuple<Dimensions<2>, std::size_t, Dimensions<1>>, M2x>);
  static_assert(pattern_collection_for<std::tuple<Dimensions<0>, Dimensions<3>>, M03>);
  static_assert(pattern_collection_for<std::tuple<Dimensions<0>, Dimensions<3>, Dimensions<1>>, M03>);

  static_assert(not is_square_shaped(m3));
  static_assert(not is_square_shaped(m23));
  static_assert(not is_square_shaped(m234));
  static_assert(compare(*is_square_shaped(m1), Dimensions<1>{}));
  static_assert(compare(*is_square_shaped(m11), Dimensions<1>{}));
  static_assert(compare(*is_square_shaped(m111), Dimensions<1>{}));
  static_assert(compare(*is_square_shaped(m22), Dimensions<2>{}));
  static_assert(compare(*is_square_shaped(m22), Dimensions<2>{}));
  static_assert(compare(*is_square_shaped(mxx_22), Dimensions<2>{}));
  static_assert(compare(*is_square_shaped(m2x_2), Dimensions<2>{}));
  static_assert(compare(*is_square_shaped(mx2_2), Dimensions<2>{}));
  static_assert(compare(*is_square_shaped(m222), Dimensions<2>{}));

  static_assert(not square_shaped<M3, values::unbounded_size, applicability::permitted>);
  static_assert(not square_shaped<M23, values::unbounded_size, applicability::permitted>);
  static_assert(not square_shaped<M234, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<M1>);
  static_assert(square_shaped<M11>);
  static_assert(square_shaped<M111>);
  static_assert(square_shaped<M22>);
  static_assert(not square_shaped<Mxx>);
  static_assert(not square_shaped<M2x>);
  static_assert(not square_shaped<Mx2>);
  static_assert(square_shaped<Mxx, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<M2x, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<Mx2, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<M222>);
  static_assert(not square_shaped<M234, values::unbounded_size, applicability::permitted>);

  static_assert(not is_one_dimensional(m3));
  static_assert(not is_one_dimensional(m23));
  static_assert(not is_one_dimensional(m234));
  static_assert(is_one_dimensional(m1));
  static_assert(is_one_dimensional(m11));
  static_assert(is_one_dimensional(m111));
  static_assert(not is_one_dimensional(m22));
  static_assert(not is_one_dimensional(m222));

  static_assert(not one_dimensional<M3, values::unbounded_size, applicability::permitted>);
  static_assert(not one_dimensional<M23, values::unbounded_size, applicability::permitted>);
  static_assert(not one_dimensional<M234, values::unbounded_size, applicability::permitted>);
  static_assert(one_dimensional<M1>);
  static_assert(one_dimensional<M11>);
  static_assert(one_dimensional<M111>);
  static_assert(not one_dimensional<Mx>);
  static_assert(one_dimensional<Mx, values::unbounded_size, applicability::permitted>);
  static_assert(not one_dimensional<Mxx>);
  static_assert(one_dimensional<Mxx, values::unbounded_size, applicability::permitted>);
  static_assert(not one_dimensional<M2x, values::unbounded_size, applicability::permitted>);
  static_assert(one_dimensional<Z1>);
  static_assert(one_dimensional<Z11>);
  static_assert(one_dimensional<Z111>);

  static_assert(not empty_object<M3>);
  static_assert(not empty_object<Mx>);
  static_assert(empty_object<M0>);
  static_assert(empty_object<M01>);
  static_assert(empty_object<M011>);
  static_assert(empty_object<M03>);
  static_assert(empty_object<M20>);
  static_assert(empty_object<M034>);
  static_assert(empty_object<M204>);
  static_assert(empty_object<M004>);

  static_assert(is_vector<0>(m3));
  static_assert(not is_vector<1>(m3));
  static_assert(not is_vector<0>(m23));
  static_assert(not is_vector<1>(m23));
  static_assert(not is_vector<0>(m234));
  static_assert(not is_vector<1>(m234));
  static_assert(not is_vector<2>(m234));
  static_assert(is_vector<0>(m1));
  static_assert(is_vector<1>(m1));
  static_assert(is_vector<2>(m1));
  static_assert(is_vector<0>(m11));
  static_assert(is_vector<1>(m11));
  static_assert(is_vector<2>(m11));
  static_assert(is_vector<0>(m111));
  static_assert(is_vector<1>(m111));
  static_assert(is_vector<2>(m111));
  static_assert(is_vector<0>(m31));
  static_assert(not is_vector<1>(m31));
  static_assert(not is_vector<0>(m13));
  static_assert(is_vector<1>(m13));
  static_assert(not is_vector<0>(m131));
  static_assert(is_vector<1>(m131));
  static_assert(not is_vector<2>(m131));
  static_assert(not is_vector<0>(m113));
  static_assert(not is_vector<1>(m113));
  static_assert(is_vector<2>(m113));

  static_assert(vector<M3, 0>);
  static_assert(not vector<M3, 1, applicability::permitted>);
  static_assert(not vector<M23, 0, applicability::permitted>);
  static_assert(not vector<M23, 1, applicability::permitted>);
  static_assert(not vector<M234, 0, applicability::permitted>);
  static_assert(not vector<M234, 1, applicability::permitted>);
  static_assert(not vector<M234, 2, applicability::permitted>);
  static_assert(vector<M1, 0>);
  static_assert(vector<M1, 1>);
  static_assert(vector<M11, 0>);
  static_assert(vector<M11, 1>);
  static_assert(vector<M31, 0>);
  static_assert(not vector<M31, 1, applicability::permitted>);
  static_assert(not vector<M13, 0, applicability::permitted>);
  static_assert(vector<M13, 1>);
  static_assert(vector<M111, 0>);
  static_assert(vector<M111, 1>);
  static_assert(vector<M111, 2>);
  static_assert(vector<M111, 3>);
  static_assert(vector<M111, 0>);
  static_assert(not vector<M131, 0, applicability::permitted>);
  static_assert(vector<M131, 1>);
  static_assert(not vector<M131, 2, applicability::permitted>);
  static_assert(not vector<M113, 0, applicability::permitted>);
  static_assert(not vector<M113, 1, applicability::permitted>);
  static_assert(vector<M113, 2>);
}

#include "linear-algebra/concepts/index_collection_for.hpp"
#include "linear-algebra/traits/access.hpp"
#include "linear-algebra/traits/access_at.hpp"

TEST(stl_interfaces, mdspan_indices)
{
  static_assert(index_collection_for<std::array<std::size_t, 0>, M1>);
  static_assert(index_collection_for<std::array<std::size_t, 1>, M1>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, M1>);
  static_assert(index_collection_for<std::vector<std::size_t>, M1>);
  static_assert(index_collection_for<std::tuple<N0>, M1>);
  static_assert(index_collection_for<std::tuple<N0, N0>, M1>);
  static_assert(not index_collection_for<std::tuple<N1>, M1>);

  static_assert(index_collection_for<std::array<std::size_t, 0>, M11>);
  static_assert(index_collection_for<std::array<std::size_t, 1>, M11>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, M11>);
  static_assert(index_collection_for<std::array<std::size_t, 3>, M11>);
  static_assert(index_collection_for<std::tuple<N0, N0>, M11>);
  static_assert(index_collection_for<std::tuple<N0, N0, N0>, M11>);
  static_assert(not index_collection_for<std::tuple<N1, N0>, M11>);
  static_assert(not index_collection_for<std::tuple<N0, N1>, M11>);

  static_assert(index_collection_for<std::array<std::size_t, 0>, M111>);
  static_assert(index_collection_for<std::array<std::size_t, 1>, M111>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, M111>);
  static_assert(index_collection_for<std::array<std::size_t, 3>, M111>);
  static_assert(index_collection_for<std::array<std::size_t, 4>, M111>);
  static_assert(index_collection_for<std::tuple<N0, N0, N0>, M111>);
  static_assert(index_collection_for<std::tuple<N0, N0, N0, N0>, M111>);
  static_assert(not index_collection_for<std::tuple<N1, N0, N0>, M111>);
  static_assert(not index_collection_for<std::tuple<N0, N1, N0>, M111>);
  static_assert(not index_collection_for<std::tuple<N0, N0, N1>, M111>);
  static_assert(not index_collection_for<std::tuple<N0, N0, N0, N1>, M111>);

  static_assert(not index_collection_for<std::array<std::size_t, 1>, M23>);
  static_assert(index_collection_for<std::array<std::size_t, 2>, M23>);
  static_assert(index_collection_for<std::array<std::size_t, 3>, M23>);
  static_assert(index_collection_for<std::tuple<N0, N0>, M23>);
  static_assert(index_collection_for<std::tuple<N1, N2>, M23>);
  static_assert(index_collection_for<std::tuple<N1, N2, N0>, M23>);
  static_assert(not index_collection_for<std::tuple<N2, N2>, M23>);
  static_assert(not index_collection_for<std::tuple<N1, N3>, M23>);

  EXPECT_EQ(access(m1, std::array{0U,0U}), 7);
  EXPECT_EQ(access(m1, std::array{0U,0U,0U}), 7);
  EXPECT_EQ(access(m11, std::array{0U,0U,0U}), 7);
  EXPECT_EQ(access(m111, std::array{0U,0U,0U}), 7);
  EXPECT_EQ(access(m3, std::array{0U}), 1);
  EXPECT_EQ(access(m3, std::array{1U}), 2);
  EXPECT_EQ(access(m3, std::array{2U}), 3);
  EXPECT_EQ(access(m3, std::array{0U,0U}), 1);
  EXPECT_EQ(access(m3, std::array{1U,0U}), 2);
  EXPECT_EQ(access(m3, std::array{2U,0U}), 3);
  EXPECT_EQ(access(m23, std::array{0U,0U}), 1);
  EXPECT_EQ(access(m23, std::array{0U,1U}), 2);
  EXPECT_EQ(access(m23, std::array{0U,2U}), 3);
  EXPECT_EQ(access(m23, std::array{1U,0U}), 4);
  EXPECT_EQ(access(m23, std::array{1U,1U}), 5);
  EXPECT_EQ(access(m23, std::array{1U,2U}), 6);
  EXPECT_EQ(access(m23c, std::array{0U,0U}), 1);
  EXPECT_EQ(access(m23c, std::array{0U,1U}), 2);
  EXPECT_EQ(access(m23c, std::array{0U,2U}), 3);
  EXPECT_EQ(access(m23c, std::array{1U,0U}), 4);
  EXPECT_EQ(access(m23c, std::array{1U,1U}), 5);
  EXPECT_EQ(access(m23c, std::array{1U,2U}), 6);

  EXPECT_EQ(access(m1, std::tuple{N0{},N0{}}), 7);
  EXPECT_EQ(access(m1, std::tuple{N0{},N0{},N0{}}), 7);
  EXPECT_EQ(access(m11, std::tuple{N0{},N0{},N0{}}), 7);
  EXPECT_EQ(access(m111, std::tuple{N0{},N0{},N0{}}), 7);
  EXPECT_EQ(access(m3, std::tuple{N0{}}), 1);
  EXPECT_EQ(access(m3, std::tuple{N1{}}), 2);
  EXPECT_EQ(access(m3, std::tuple{N2{}}), 3);
  EXPECT_EQ(access(m3, std::tuple{N0{},N0{}}), 1);
  EXPECT_EQ(access(m3, std::tuple{N1{},N0{}}), 2);
  EXPECT_EQ(access(m3, std::tuple{N2{},N0{}}), 3);
  EXPECT_EQ(access(m23, std::tuple{N0{},N0{}}), 1);
  EXPECT_EQ(access(m23, std::tuple{N0{},N1{}}), 2);
  EXPECT_EQ(access(m23, std::tuple{N0{},N2{}}), 3);
  EXPECT_EQ(access(m23, std::tuple{N1{},N0{}}), 4);
  EXPECT_EQ(access(m23, std::tuple{N1{},N1{}}), 5);
  EXPECT_EQ(access(m23, std::tuple{N1{},N2{}}), 6);

  EXPECT_EQ(access(m1, std::vector{0U,0U}), 7);
  EXPECT_EQ(access(m1, std::vector{0U,0U,0U}), 7);
  EXPECT_EQ(access(m11, std::vector{0U,0U,0U}), 7);
  EXPECT_EQ(access(m111, std::vector{0U,0U,0U}), 7);
  EXPECT_EQ(access(m3, std::vector{0U}), 1);
  EXPECT_EQ(access(m3, std::vector{1U}), 2);
  EXPECT_EQ(access(m3, std::vector{2U}), 3);
  EXPECT_EQ(access(m3, std::vector{0U,0U}), 1);
  EXPECT_EQ(access(m3, std::vector{1U,0U}), 2);
  EXPECT_EQ(access(m3, std::vector{2U,0U}), 3);
  EXPECT_EQ(access(m23, std::vector{0U,0U}), 1);
  EXPECT_EQ(access(m23, std::vector{0U,1U}), 2);
  EXPECT_EQ(access(m23, std::vector{0U,2U}), 3);
  EXPECT_EQ(access(m23, std::vector{1U,0U}), 4);
  EXPECT_EQ(access(m23, std::vector{1U,1U}), 5);
  EXPECT_EQ(access(m23, std::vector{1U,2U}), 6);

  EXPECT_EQ(access(m1, 0U, 0U), 7);
  EXPECT_EQ(access(m1, 0U, 0U, 0U), 7);
  EXPECT_EQ(access(m11, 0U, 0U, 0U), 7);
  EXPECT_EQ(access(m111, 0U, 0U, 0U), 7);
  EXPECT_EQ(access(m3, 0U), 1);
  EXPECT_EQ(access(m3, 1U), 2);
  EXPECT_EQ(access(m3, 2U), 3);
  EXPECT_EQ(access(m3, 0U, 0U), 1);
  EXPECT_EQ(access(m3, 1U, 0U), 2);
  EXPECT_EQ(access(m3, 2U, 0U), 3);
  EXPECT_EQ(access(m23, 0U, 0U), 1);
  EXPECT_EQ(access(m23, 0U, 1U), 2);
  EXPECT_EQ(access(m23, 0U, 2U), 3);
  EXPECT_EQ(access(m23, 1U, 0U), 4);
  EXPECT_EQ(access(m23, 1U, 1U), 5);
  EXPECT_EQ(access(m23, 1U, 2U), 6);
  EXPECT_EQ(access(mxx_23, 0U, 0U), 1);
  EXPECT_EQ(access(mxx_23, 0U, 1U), 2);
  EXPECT_EQ(access(mxx_23, 0U, 2U), 3);
  EXPECT_EQ(access(m2x_3, 0U, 0U), 1);
  EXPECT_EQ(access(m2x_3, 0U, 1U), 2);
  EXPECT_EQ(access(m2x_3, 0U, 2U), 3);
  EXPECT_EQ(access(mx3_2, 0U, 0U), 1);
  EXPECT_EQ(access(mx3_2, 0U, 1U), 2);
  EXPECT_EQ(access(mx3_2, 0U, 2U), 3);

  EXPECT_EQ(access_at(m1, 0U), 7);
  EXPECT_EQ(access_at(m1, 0U, 0U), 7);
  EXPECT_EQ(access_at(m1, std::vector{0U, 0U}), 7);
  EXPECT_ANY_THROW(access_at(m1, 1U));
  EXPECT_ANY_THROW(access_at(m1, 0U, 1U));
  EXPECT_ANY_THROW(access_at(m1, std::vector{0U, 1U}));
  EXPECT_EQ(access_at(m11, 0U, 0U), 7);
  EXPECT_EQ(access_at(m11, 0U, 0U, 0U), 7);
  EXPECT_EQ(access_at(m11, std::vector{0U, 0U, 0U}), 7);
  EXPECT_ANY_THROW(access_at(m11, 0U, 1U));
  EXPECT_ANY_THROW(access_at(m11, 0U, 0U, 1U));
  EXPECT_ANY_THROW(access_at(m11, std::vector{0U, 0U, 1U}));
  EXPECT_EQ(access_at(m111, 0U, 0U, 0U), 7);
  EXPECT_EQ(access_at(m111, 0U, 0U, 0U, 0U), 7);
  EXPECT_EQ(access_at(m111, std::vector{0U, 0U, 0U, 0U, 0U}), 7);
  EXPECT_ANY_THROW(access_at(m111, 0U, 0U, 1U));
  EXPECT_ANY_THROW(access_at(m111, 0U, 0U, 0U, 1U));
  EXPECT_ANY_THROW(access_at(m111, std::vector{0U, 0U, 0U, 1U}));
  EXPECT_EQ(access_at(m3, 0U), 1);
  EXPECT_EQ(access_at(m3, 1U), 2);
  EXPECT_EQ(access_at(m3, 2U), 3);
  EXPECT_ANY_THROW(access_at(m3, 3U));
  EXPECT_EQ(access_at(m3, 0U, 0U), 1);
  EXPECT_EQ(access_at(m3, 1U, 0U), 2);
  EXPECT_EQ(access_at(m3, 2U, 0U), 3);
  EXPECT_ANY_THROW(access_at(m3, 2U, 1U));
  EXPECT_ANY_THROW(access_at(m3, 3U, 0U));
  EXPECT_EQ(access_at(m23, 0U, 0U), 1);
  EXPECT_EQ(access_at(m23, 0U, 1U), 2);
  EXPECT_EQ(access_at(m23, 0U, 2U), 3);
  EXPECT_EQ(access_at(m23, 1U, 0U), 4);
  EXPECT_EQ(access_at(m23, 1U, 1U), 5);
  EXPECT_EQ(access_at(m23, 1U, 2U), 6);
  EXPECT_ANY_THROW(access_at(m23, 2U, 2U));
  EXPECT_ANY_THROW(access_at(m23, 1U, 3U));
  EXPECT_ANY_THROW(access_at(m23, 1U, 2U, 1U));

  auto a23c2 = std::array{11., 12., 13., 14., 15., 16.};
  M23 m23c2 {a23c2.data()};
  M23 m23w = m23;
  access(m23w, 0U, 0U) = 11;
  access(m23w, 0U, 1U) = 12;
  access(m23w, 0U, 2U) = 13;
  access(m23w, 1U, 0U) = 14;
  access(m23w, 1U, 1U) = 15;
  access(m23w, 1U, 2U) = 16;
  EXPECT_TRUE(is_near(m23w, m23c2));

  access_at(m23w, 0U, 0U) = 1;
  access_at(m23w, 0U, 1U) = 2;
  access_at(m23w, 0U, 2U) = 3;
  access_at(m23w, 1U, 0U) = 4;
  access_at(m23w, 1U, 1U) = 5;
  access_at(m23w, 1U, 2U) = 6;
  EXPECT_ANY_THROW(access_at(m23w, 2U, 2U) = 6);
  EXPECT_TRUE(is_near(m23w, m23c));
}

#include "linear-algebra/concepts/zero.hpp"
#include "linear-algebra/traits/triangle_type_of.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
//#include "linear-algebra/traits/hermitian_adapter_type_of.hpp"

TEST(stl_interfaces, mdspan_special_matrices)
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

  static_assert(triangle_type_of_v<M1> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<M11> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<M111> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<M3> == triangle_type::none);
  static_assert(triangle_type_of_v<M23> == triangle_type::none);
  static_assert(triangle_type_of_v<Mxx> == triangle_type::none);
  static_assert(triangle_type_of_v<M234> == triangle_type::none);

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

  static_assert(triangular_matrix<M1, triangle_type::diagonal>);
  static_assert(triangular_matrix<M1, triangle_type::upper>);
  static_assert(triangular_matrix<M1, triangle_type::lower>);
  static_assert(triangular_matrix<M11, triangle_type::diagonal>);
  static_assert(triangular_matrix<M11, triangle_type::upper>);
  static_assert(triangular_matrix<M11, triangle_type::lower>);
  static_assert(triangular_matrix<M111, triangle_type::diagonal>);
  static_assert(triangular_matrix<M111, triangle_type::upper>);
  static_assert(triangular_matrix<M111, triangle_type::lower>);
  static_assert(not triangular_matrix<M234>);
  static_assert(not triangular_matrix<Mx>);
  static_assert(triangular_matrix<I1, triangle_type::diagonal>);
  static_assert(triangular_matrix<I11, triangle_type::diagonal>);
  static_assert(triangular_matrix<I111, triangle_type::diagonal>);

  static_assert(diagonal_matrix<Z1>);
  static_assert(diagonal_matrix<Z11>);
  static_assert(diagonal_matrix<Z111>);
  static_assert(diagonal_matrix<Z3>);
  static_assert(diagonal_matrix<Z23>);
  static_assert(diagonal_matrix<Z234>);
  static_assert(diagonal_matrix<M1>);
  static_assert(diagonal_matrix<M11>);
  static_assert(diagonal_matrix<M111>);
  static_assert(not diagonal_matrix<M234>);
  static_assert(diagonal_matrix<I1>);
  static_assert(diagonal_matrix<I11>);
  static_assert(diagonal_matrix<I111>);

  static_assert(hermitian_matrix<Z1>);
  static_assert(hermitian_matrix<Z11>);
  static_assert(hermitian_matrix<Z111>);
  static_assert(not hermitian_matrix<Z3>);
  static_assert(not hermitian_matrix<Z23>);
  static_assert(not hermitian_matrix<Z234>);
  static_assert(hermitian_matrix<M1>);
  static_assert(hermitian_matrix<M11>);
  static_assert(hermitian_matrix<M111>);
  static_assert(not hermitian_matrix<M234>);
  static_assert(hermitian_matrix<I1>);
  static_assert(hermitian_matrix<I11>);
  static_assert(hermitian_matrix<I111>);
}

#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/traits/constant_value_of.hpp"
#include "linear-algebra/concepts/identity_matrix.hpp"

TEST(stl_interfaces, mdspan_constants)
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
  static_assert(constant_value_of_v<decltype(z1)> == 0);
  static_assert(constant_value_of_v<decltype(z11)> == 0);
  static_assert(constant_value_of_v<decltype(z111)> == 0);

  static_assert(constant_object<M1>);
  static_assert(constant_object<M11>);
  static_assert(constant_object<M111>);
  static_assert(not constant_object<Mx>);
  static_assert(constant_diagonal_object<M1>);
  static_assert(constant_diagonal_object<M11>);
  static_assert(constant_diagonal_object<M111>);
  static_assert(not constant_diagonal_object<Mx>);

  static_assert(constant_object<Z3>);
  static_assert(constant_object<Z23>);
  static_assert(constant_object<Z234>);
  static_assert(constant_diagonal_object<Z3>);
  static_assert(constant_diagonal_object<Z23>);
  static_assert(constant_diagonal_object<Z234>);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z3))> == 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z23))> == 0);
  static_assert(values::fixed_value_of_v<decltype(constant_value(z234))> == 0);
  static_assert(constant_value_of_v<decltype(z3)> == 0);
  static_assert(constant_value_of_v<decltype(z23)> == 0);
  static_assert(constant_value_of_v<decltype(z234)> == 0);

  static_assert(identity_matrix<I1>);
  static_assert(identity_matrix<I11>);
  static_assert(identity_matrix<I111>);
  static_assert(not identity_matrix<Z11>);
  static_assert(not identity_matrix<M11>);
}
