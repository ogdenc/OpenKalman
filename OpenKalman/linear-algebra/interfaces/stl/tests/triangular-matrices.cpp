/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests relating to triangular_adapter.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"

#include "linear-algebra/adapters/triangular_adapter.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;
using namespace OpenKalman::patterns;


namespace
{
  using cdouble = std::complex<double>;

  using M1 = stdex::mdspan<double, stdex::extents<std::size_t, 1>>;
  using M11 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 1>>;
  using M1x = stdex::mdspan<double, stdex::extents<std::size_t, 1, stdex::dynamic_extent>>;
  using Mx1 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 1>>;
  using Mxx = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>>;

  using M2 = stdex::mdspan<double, stdex::extents<std::size_t, 2>>;
  using M21 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 1>>;
  using M12 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 2>>;
  using M22 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 2>>;
  using M2x = stdex::mdspan<double, stdex::extents<std::size_t, 2, stdex::dynamic_extent>>;
  using Mx2 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>;

  using CM22 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 2, 2>>;

  template<typename T> using TL = triangular_adapter<T, triangle_type::lower>;
  template<typename T> using TU = triangular_adapter<T, triangle_type::upper>;
  template<typename T> using TD = triangular_adapter<T, triangle_type::diagonal>;

  template<typename N, triangle_type tri>
  using MT = stdex::mdspan<
    typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::element_type,
    typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::extents_type,
    interface::layout_triangle_partition<typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::layout_type, tri>,
    interface::triangular_accessor<typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::accessor_type>>;

  template<typename N>
  using MTL = MT<N, triangle_type::lower>;

  template<typename N>
  using MTU = MT<N, triangle_type::upper>;

  template<typename N>
  using MTD = MT<N, triangle_type::diagonal>;

}

#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"

TEST(stl_interfaces, triangular_adapter_static_properties)
{
  static_assert(diagonal_matrix<TD<M1>>);
  static_assert(diagonal_matrix<TD<M11>>);
  static_assert(diagonal_matrix<TD<M1x>>);
  static_assert(diagonal_matrix<TD<Mx1>>);
  static_assert(diagonal_matrix<TD<Mxx>>);

  static_assert(diagonal_matrix<MTD<M1>>);
  static_assert(diagonal_matrix<MTD<M11>>);
  static_assert(diagonal_matrix<MTD<M1x>>);
  static_assert(diagonal_matrix<MTD<Mx1>>);
  static_assert(diagonal_matrix<MTD<Mxx>>);

  static_assert(triangle_type_of_v<TD<M2>> == triangle_type::diagonal);
  static_assert(triangle_type_of_v<MTD<M2>> == triangle_type::diagonal);

  static_assert(diagonal_matrix<TD<M2>>);
  static_assert(diagonal_matrix<TD<M21>>);
  static_assert(diagonal_matrix<TD<M12>>);
  static_assert(diagonal_matrix<TD<M22>>);
  static_assert(diagonal_matrix<TD<M2x>>);
  static_assert(diagonal_matrix<TD<Mx2>>);
  static_assert(diagonal_matrix<TD<Mxx>>);

  static_assert(diagonal_matrix<MTD<M2>>);
  static_assert(diagonal_matrix<MTD<M21>>);
  static_assert(diagonal_matrix<MTD<M12>>);
  static_assert(diagonal_matrix<MTD<M22>>);
  static_assert(diagonal_matrix<MTD<M2x>>);
  static_assert(diagonal_matrix<MTD<Mx2>>);
  static_assert(diagonal_matrix<MTD<Mxx>>);

  static_assert(not diagonal_matrix<TL<M2>>);
  static_assert(diagonal_matrix<TU<M2>>);
  static_assert(not diagonal_matrix<TL<M21>>);
  static_assert(diagonal_matrix<TU<M21>>);
  static_assert(diagonal_matrix<TL<M12>>);
  static_assert(not diagonal_matrix<TU<M12>>);

  static_assert(not diagonal_matrix<MTL<M2>>);
  static_assert(diagonal_matrix<MTU<M2>>);
  static_assert(not diagonal_matrix<MTL<M21>>);
  static_assert(diagonal_matrix<MTU<M21>>);
  static_assert(diagonal_matrix<MTL<M12>>);
  static_assert(not diagonal_matrix<MTU<M12>>);

  static_assert(diagonal_matrix<TU<TL<M22>>>);
  static_assert(diagonal_matrix<TL<TU<M22>>>);
  static_assert(diagonal_matrix<TL<TD<M22>>>);
  static_assert(diagonal_matrix<TU<TD<M22>>>);
  static_assert(diagonal_matrix<TD<TL<M22>>>);
  static_assert(diagonal_matrix<TD<TU<M22>>>);
  static_assert(diagonal_matrix<TD<TD<M22>>>);
  static_assert(diagonal_matrix<TD<TD<CM22>>>);
  static_assert(not hermitian_matrix<TD<TD<CM22>>>);

  static_assert(diagonal_matrix<MTU<MTL<M22>>>);
  static_assert(diagonal_matrix<MTL<MTU<M22>>>);
  static_assert(diagonal_matrix<MTL<MTD<M22>>>);
  static_assert(diagonal_matrix<MTU<MTD<M22>>>);
  static_assert(diagonal_matrix<MTD<MTL<M22>>>);
  static_assert(diagonal_matrix<MTD<MTU<M22>>>);
  static_assert(diagonal_matrix<MTD<MTD<M22>>>);
  static_assert(diagonal_matrix<MTD<MTD<CM22>>>);
  static_assert(not hermitian_matrix<MTD<MTD<CM22>>>);

  static_assert(not diagonal_matrix<TL<M22>>);
  static_assert(not diagonal_matrix<TU<M22>>);
  static_assert(not diagonal_matrix<TL<M2x>>);
  static_assert(not diagonal_matrix<TU<M2x>>);
  static_assert(not diagonal_matrix<TL<Mx2>>);
  static_assert(not diagonal_matrix<TU<Mx2>>);
  static_assert(not diagonal_matrix<TL<Mxx>>);
  static_assert(not diagonal_matrix<TU<Mxx>>);
  static_assert(not diagonal_matrix<TL<CM22>>);
  static_assert(not diagonal_matrix<TU<CM22>>);
  static_assert(not hermitian_matrix<TU<CM22>>);

  static_assert(not diagonal_matrix<MTL<M22>>);
  static_assert(not diagonal_matrix<MTU<M22>>);
  static_assert(not diagonal_matrix<MTL<M2x>>);
  static_assert(not diagonal_matrix<MTU<M2x>>);
  static_assert(not diagonal_matrix<MTL<Mx2>>);
  static_assert(not diagonal_matrix<MTU<Mx2>>);
  static_assert(not diagonal_matrix<MTL<Mxx>>);
  static_assert(not diagonal_matrix<MTU<Mxx>>);
  static_assert(not diagonal_matrix<MTL<CM22>>);
  static_assert(not diagonal_matrix<MTU<CM22>>);
  static_assert(not hermitian_matrix<MTU<CM22>>);

  static_assert(triangular_matrix<TL<M22>, triangle_type::lower>);
  static_assert(triangular_matrix<TL<M2x>, triangle_type::lower>);
  static_assert(triangular_matrix<TL<Mx2>, triangle_type::lower>);
  static_assert(triangular_matrix<TL<Mxx>, triangle_type::lower>);
  static_assert(triangular_matrix<TL<CM22>, triangle_type::lower>);
  static_assert(not triangular_matrix<TU<M22>, triangle_type::lower>);
  static_assert(not triangular_matrix<TU<M2x>, triangle_type::lower>);
  static_assert(not triangular_matrix<TU<Mx2>, triangle_type::lower>);
  static_assert(not triangular_matrix<TU<Mxx>, triangle_type::lower>);
  static_assert(not triangular_matrix<TU<CM22>, triangle_type::lower>);

  static_assert(triangular_matrix<TU<M22>, triangle_type::upper>);
  static_assert(triangular_matrix<TU<M2x>, triangle_type::upper>);
  static_assert(triangular_matrix<TU<Mx2>, triangle_type::upper>);
  static_assert(triangular_matrix<TU<Mxx>, triangle_type::upper>);
  static_assert(triangular_matrix<TU<CM22>, triangle_type::upper>);
  static_assert(not triangular_matrix<TL<M22>, triangle_type::upper>);
  static_assert(not triangular_matrix<TL<M2x>, triangle_type::upper>);
  static_assert(not triangular_matrix<TL<Mx2>, triangle_type::upper>);
  static_assert(not triangular_matrix<TL<Mxx>, triangle_type::upper>);
  static_assert(not triangular_matrix<TL<CM22>, triangle_type::upper>);

  static_assert(triangular_matrix<MTL<M22>, triangle_type::lower>);
  static_assert(triangular_matrix<MTL<M2x>, triangle_type::lower>);
  static_assert(triangular_matrix<MTL<Mx2>, triangle_type::lower>);
  static_assert(triangular_matrix<MTL<Mxx>, triangle_type::lower>);
  static_assert(triangular_matrix<MTL<CM22>, triangle_type::lower>);
  static_assert(not triangular_matrix<MTU<M22>, triangle_type::lower>);
  static_assert(not triangular_matrix<MTU<M2x>, triangle_type::lower>);
  static_assert(not triangular_matrix<MTU<Mx2>, triangle_type::lower>);
  static_assert(not triangular_matrix<MTU<Mxx>, triangle_type::lower>);
  static_assert(not triangular_matrix<MTU<CM22>, triangle_type::lower>);

  static_assert(triangular_matrix<MTU<M22>, triangle_type::upper>);
  static_assert(triangular_matrix<MTU<M2x>, triangle_type::upper>);
  static_assert(triangular_matrix<MTU<Mx2>, triangle_type::upper>);
  static_assert(triangular_matrix<MTU<Mxx>, triangle_type::upper>);
  static_assert(triangular_matrix<MTU<CM22>, triangle_type::upper>);
  static_assert(not triangular_matrix<MTL<M22>, triangle_type::upper>);
  static_assert(not triangular_matrix<MTL<M2x>, triangle_type::upper>);
  static_assert(not triangular_matrix<MTL<Mx2>, triangle_type::upper>);
  static_assert(not triangular_matrix<MTL<Mxx>, triangle_type::upper>);
  static_assert(not triangular_matrix<MTL<CM22>, triangle_type::upper>);

  static_assert(square_shaped<TD<M22>>);
  static_assert(square_shaped<TL<M22>>);
  static_assert(square_shaped<TU<M22>>);
  static_assert(square_shaped<TD<M2x>, 2, applicability::permitted>);
  static_assert(square_shaped<TL<M2x>, 2, applicability::permitted>);
  static_assert(square_shaped<TU<M2x>, 2, applicability::permitted>);
  static_assert(square_shaped<TD<Mx2>, 2, applicability::permitted>);
  static_assert(square_shaped<TL<Mx2>, 2, applicability::permitted>);
  static_assert(square_shaped<TU<Mx2>, 2, applicability::permitted>);
  static_assert(square_shaped<TD<Mxx>, 2, applicability::permitted>);
  static_assert(square_shaped<TL<Mxx>, 2, applicability::permitted>);
  static_assert(square_shaped<TU<Mxx>, 2, applicability::permitted>);
  static_assert(square_shaped<TD<CM22>>);
  static_assert(square_shaped<TL<CM22>>);
  static_assert(square_shaped<TU<CM22>>);

  static_assert(square_shaped<MTD<M22>>);
  static_assert(square_shaped<MTL<M22>>);
  static_assert(square_shaped<MTU<M22>>);
  static_assert(square_shaped<MTD<M2x>, 2, applicability::permitted>);
  static_assert(square_shaped<MTL<M2x>, 2, applicability::permitted>);
  static_assert(square_shaped<MTU<M2x>, 2, applicability::permitted>);
  static_assert(square_shaped<MTD<Mx2>, 2, applicability::permitted>);
  static_assert(square_shaped<MTL<Mx2>, 2, applicability::permitted>);
  static_assert(square_shaped<MTU<Mx2>, 2, applicability::permitted>);
  static_assert(square_shaped<MTD<Mxx>, 2, applicability::permitted>);
  static_assert(square_shaped<MTL<Mxx>, 2, applicability::permitted>);
  static_assert(square_shaped<MTU<Mxx>, 2, applicability::permitted>);
  static_assert(square_shaped<TD<CM22>>);
  static_assert(square_shaped<MTL<CM22>>);
  static_assert(square_shaped<MTU<CM22>>);
}

#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/functions/copy_from.hpp"
#include "linear-algebra/functions/make_constant.hpp"
#include "linear-algebra/functions/make_zero.hpp"
#include "linear-algebra/functions/to_triangular.hpp"

TEST(stl_interfaces, triangular_adapter_dynamic_properties)
{
  using A22 = double[2][2];
  A22 a22 {{1, 2}, {3, 4}};

  decltype(auto) la22 = to_triangular<triangle_type::lower>(a22);
  EXPECT_EQ((la22[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((la22[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((la22[std::array{1U, 0U}]), 3.);
  EXPECT_EQ((la22[std::array{1U, 1U}]), 4.);

  decltype(auto) ua22 = to_triangular<triangle_type::upper>(a22);
  EXPECT_EQ((ua22[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((ua22[std::array{0U, 1U}]), 2.);
  EXPECT_EQ((ua22[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((ua22[std::array{1U, 1U}]), 4.);

  decltype(auto) da22 = to_triangular<triangle_type::diagonal>(a22);
  EXPECT_EQ((da22[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((da22[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((da22[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((da22[std::array{1U, 1U}]), 4.);

  auto pa22 = attach_patterns(a22, std::tuple{Polar{}, Dimensions<2>{}});
  decltype(auto) pla22 = to_triangular<triangle_type::lower>(pa22);
  static_assert(compares_with_pattern_collection<decltype(pla22), std::tuple<Polar<>, Dimensions<2>>, &stdex::is_eq, applicability::guaranteed>);

  auto a6 = std::array {1., 2., 3., 4., 5., 6.};

  auto ma23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>(a6.data());

  decltype(auto) la23 = to_triangular<triangle_type::lower>(ma23);
  static_assert(triangular_matrix<decltype(la23), triangle_type::lower>);
  EXPECT_EQ((la23[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((la23[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((la23[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((la23[std::array{1U, 0U}]), 4.);
  EXPECT_EQ((la23[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((la23[std::array{1U, 2U}]), 0.);

  decltype(auto) ua23 = to_triangular<triangle_type::upper>(ma23);
  static_assert(triangular_matrix<decltype(ua23), triangle_type::upper>);
  EXPECT_EQ((ua23[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((ua23[std::array{0U, 1U}]), 2.);
  EXPECT_EQ((ua23[std::array{0U, 2U}]), 3.);
  EXPECT_EQ((ua23[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((ua23[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((ua23[std::array{1U, 2U}]), 6.);

  decltype(auto) da23 = to_triangular<triangle_type::diagonal>(ma23);
  static_assert(triangular_matrix<decltype(da23), triangle_type::diagonal>);
  EXPECT_EQ((da23[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((da23[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((da23[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((da23[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((da23[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((da23[std::array{1U, 2U}]), 0.);

  auto ma32 = stdex::mdspan<double, stdex::extents<std::size_t, 3, 2>>(a6.data());

  decltype(auto) la32 = to_triangular<triangle_type::lower>(ma32);
  static_assert(triangular_matrix<decltype(la32), triangle_type::lower>);
  EXPECT_EQ((la32[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((la32[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((la32[std::array{1U, 0U}]), 3.);
  EXPECT_EQ((la32[std::array{1U, 1U}]), 4.);
  EXPECT_EQ((la32[std::array{2U, 0U}]), 5.);
  EXPECT_EQ((la32[std::array{2U, 1U}]), 6.);

  decltype(auto) ua32 = to_triangular<triangle_type::upper>(ma32);
  static_assert(triangular_matrix<decltype(ua32), triangle_type::upper>);
  EXPECT_EQ((ua32[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((ua32[std::array{0U, 1U}]), 2.);
  EXPECT_EQ((ua32[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((ua32[std::array{1U, 1U}]), 4.);
  EXPECT_EQ((ua32[std::array{2U, 0U}]), 0.);
  EXPECT_EQ((ua32[std::array{2U, 1U}]), 0.);

  decltype(auto) da32 = to_triangular<triangle_type::diagonal>(ma32);
  static_assert(triangular_matrix<decltype(da32), triangle_type::diagonal>);
  EXPECT_EQ((da32[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((da32[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((da32[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((da32[std::array{1U, 1U}]), 4.);
  EXPECT_EQ((da32[std::array{2U, 0U}]), 0.);
  EXPECT_EQ((da32[std::array{2U, 1U}]), 0.);

  auto maxx_23 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>>(a6.data(), 2, 3);

  decltype(auto) laxx_23 = to_triangular<triangle_type::lower>(maxx_23);
  static_assert(triangular_matrix<decltype(laxx_23), triangle_type::lower>);
  EXPECT_EQ((laxx_23[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((laxx_23[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((laxx_23[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((laxx_23[std::array{1U, 0U}]), 4.);
  EXPECT_EQ((laxx_23[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((laxx_23[std::array{1U, 2U}]), 0.);

  decltype(auto) daxx_23 = to_triangular<triangle_type::diagonal>(maxx_23);
  static_assert(triangular_matrix<decltype(daxx_23), triangle_type::diagonal>);
  EXPECT_EQ((daxx_23[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((daxx_23[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((daxx_23[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((daxx_23[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((daxx_23[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((daxx_23[std::array{1U, 2U}]), 0.);

  auto a6w = std::array {9., 9., 9., 9., 9., 9.};
  auto m23w = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>(a6w.data());
  EXPECT_FALSE(is_near(m23w, la23));
  copy_from(m23w, la23);
  EXPECT_TRUE(is_near(m23w, la23));
  copy_from(m23w, ua23);
  EXPECT_TRUE(is_near(m23w, ua23));
  copy_from(m23w, da23);
  EXPECT_TRUE(is_near(m23w, da23));
  copy_from(m23w, laxx_23);
  EXPECT_TRUE(is_near(m23w, laxx_23));
  copy_from(m23w, daxx_23);
  EXPECT_TRUE(is_near(m23w, daxx_23));

  auto lz23 = to_triangular<triangle_type::lower>(make_zero<double>(stdex::extents<std::size_t, 2, 3>{}));
  static_assert(triangular_matrix<decltype(lz23), triangle_type::lower>);
  EXPECT_EQ(constant_value(lz23), 0);
  EXPECT_EQ((lz23[std::array{0U, 0U}]), 0.);
  EXPECT_EQ((lz23[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((lz23[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((lz23[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((lz23[std::array{1U, 1U}]), 0.);
  EXPECT_EQ((lz23[std::array{1U, 2U}]), 0.);

  auto mc23_5 = make_constant(5_uz, stdex::extents<std::size_t, 2, 3>{});

  auto lc23_5 = to_triangular<triangle_type::lower>(mc23_5);
  static_assert(triangular_matrix<decltype(lc23_5), triangle_type::lower>);
  EXPECT_EQ((lc23_5[std::array{0U, 0U}]), 5.);
  EXPECT_EQ((lc23_5[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((lc23_5[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((lc23_5[std::array{1U, 0U}]), 5.);
  EXPECT_EQ((lc23_5[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((lc23_5[std::array{1U, 2U}]), 0.);

  auto uc23_5 = to_triangular<triangle_type::upper>(mc23_5);
  static_assert(triangular_matrix<decltype(uc23_5), triangle_type::upper>);
  EXPECT_EQ((uc23_5[std::array{0U, 0U}]), 5.);
  EXPECT_EQ((uc23_5[std::array{0U, 1U}]), 5.);
  EXPECT_EQ((uc23_5[std::array{0U, 2U}]), 5.);
  EXPECT_EQ((uc23_5[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((uc23_5[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((uc23_5[std::array{1U, 2U}]), 5.);

  auto dc23_5 = to_triangular<triangle_type::diagonal>(mc23_5);
  static_assert(triangular_matrix<decltype(dc23_5), triangle_type::diagonal>);
  EXPECT_EQ((dc23_5[std::array{0U, 0U}]), 5.);
  EXPECT_EQ((dc23_5[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((dc23_5[std::array{0U, 2U}]), 0.);
  EXPECT_EQ((dc23_5[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((dc23_5[std::array{1U, 1U}]), 5.);
  EXPECT_EQ((dc23_5[std::array{1U, 2U}]), 0.);
}


TEST(stl_interfaces, triangular_adapter_functions)
{
  auto a6 = std::array {9., 9., 9., 9., 9., 9.};
  auto m23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>(a6.data());
  decltype(auto) la23 = to_triangular<triangle_type::lower>(m23);

}

/*
TEST(stl_interfaces, TriangularAdapter_overloads)
{
  M22 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  //
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(TL<M22>(3., 0, 1, 3)), ml));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(TU<M22>(3., 1, 0, 3)), mu));
  //
  EXPECT_TRUE(is_near(make_self_contained(TL<M22>(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_self_contained(TU<M22>(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(TL<M22> {9, 3, 3, 10} * 2))>, TL<M22>>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(TU<M22> {9, 3, 3, 10} * 2))>, TU<M22>>);
  //
  //
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::lower>(eigen_matrix_t<double, 1, 1>(4))), eigen_matrix_t<double, 1, 1>(16)));
  static_assert(one_by_one_matrix<decltype(Cholesky_square(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::lower>(eigen_matrix_t<double, 1, 1>(4))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<decltype(M22::Identity()), triangle_type::lower>(M22::Identity())), M22::Identity()));
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<decltype(M22::Identity()), triangle_type::upper>(M22::Identity())), M22::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_square(triangular_adapter<decltype(M22::Identity()), triangle_type::lower>(M22::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_square(triangular_adapter<decltype(M22::Identity()), triangle_type::upper>(M22::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::lower>(make_zero_matrix_like<M22>())), M22::Zero()));
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::upper>(make_zero_matrix_like<M22>())), M22::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_square(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::lower>(make_zero_matrix_like<M22>())))>);
  static_assert(zero_matrix<decltype(Cholesky_square(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::upper>(make_zero_matrix_like<M22>())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<decltype(to_diagonal_adapter{2., 3}), triangle_type::lower>(to_diagonal_adapter{2., 3})), to_diagonal_adapter{4., 9}));
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<decltype(to_diagonal_adapter{2., 3}), triangle_type::upper>(to_diagonal_adapter{2., 3})), to_diagonal_adapter{4., 9}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_square(triangular_adapter<decltype(to_diagonal_adapter{2., 3}), triangle_type::lower>(to_diagonal_adapter{2., 3})))>);
  static_assert(internal::diagonal_expr<decltype(Cholesky_square(triangular_adapter<decltype(to_diagonal_adapter{2., 3}), triangle_type::upper>(to_diagonal_adapter{2., 3})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<M22, triangle_type::diagonal>(ml)), to_diagonal_adapter{9., 9}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_square(triangular_adapter<M22, triangle_type::diagonal>(ml)))>);
  //
  TL<M22> lower {3., 0, 1, 3};
  EXPECT_TRUE(is_near(Cholesky_square(lower), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(TL<M22> {3., 0, 1, 3}), mat22(9., 3, 3, 10)));
  static_assert(hermitian_adapter_concept<decltype(Cholesky_square(TL<M22> {3, 0, 1, 3})), triangle_type::lower>);
  //
  TU<M22> upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(Cholesky_square(upper), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(TU<M22> {3., 1, 0, 3}), mat22(9., 3, 3, 10)));
  static_assert(hermitian_adapter_concept<decltype(Cholesky_square(TU<M22> {3, 1, 0, 3})), triangle_type::upper>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_square(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  //
  //
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::lower>(eigen_matrix_t<double, 1, 1>(4))), eigen_matrix_t<double, 1, 1>(2)));
  static_assert(one_by_one_matrix<decltype(Cholesky_factor(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::lower>(eigen_matrix_t<double, 1, 1>(4))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<decltype(M22::Identity()), triangle_type::lower>(M22::Identity())), M22::Identity()));
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<decltype(M22::Identity()), triangle_type::upper>(M22::Identity())), M22::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_factor(triangular_adapter<decltype(M22::Identity()), triangle_type::lower>(M22::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_factor(triangular_adapter<decltype(M22::Identity()), triangle_type::upper>(M22::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::lower>(make_zero_matrix_like<M22>())), M22::Zero()));
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::upper>(make_zero_matrix_like<M22>())), M22::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_factor(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::lower>(make_zero_matrix_like<M22>())))>);
  static_assert(zero_matrix<decltype(Cholesky_factor(triangular_adapter<decltype(make_zero_matrix_like<M22>()), triangle_type::upper>(make_zero_matrix_like<M22>())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<decltype(to_diagonal_adapter{4., 9}), triangle_type::lower>(to_diagonal_adapter{4., 9})), to_diagonal_adapter{2., 3}));
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<decltype(to_diagonal_adapter{4., 9}), triangle_type::upper>(to_diagonal_adapter{4., 9})), to_diagonal_adapter{2., 3}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_factor(triangular_adapter<decltype(to_diagonal_adapter{4., 9}), triangle_type::lower>(to_diagonal_adapter{4., 9})))>);
  static_assert(internal::diagonal_expr<decltype(Cholesky_factor(triangular_adapter<decltype(to_diagonal_adapter{4., 9}), triangle_type::upper>(to_diagonal_adapter{4., 9})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<M22, triangle_type::diagonal>(ml)), to_diagonal_adapter{std::sqrt(3.), std::sqrt(3.)}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_factor(triangular_adapter<M22, triangle_type::diagonal>(ml)))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(Cholesky_factor(triangular_adapter<eigen_matrix_t<double, 1, 1>, triangle_type::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  //
  //
  EXPECT_TRUE(is_near(diagonal_of(TL<M22> {3., 0, 1, 3}), make_eigen_matrix<double, 2, 1>(3, 3)));
  EXPECT_TRUE(is_near(diagonal_of(TU<M22> {3., 1, 0, 3}), make_eigen_matrix<double, 2, 1>(3, 3)));
  //
  EXPECT_TRUE(is_near(transpose(TL<M22> {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(transpose(TU<M22> {3., 1, 0, 3}), ml));
  EXPECT_TRUE(is_near(transpose(TL<CM22> {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}), TU<CM22> {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}));
  EXPECT_TRUE(is_near(transpose(TU<CM22> {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}), TL<CM22> {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}));
  //
  EXPECT_TRUE(is_near(conjugate_transpose(TL<M22> {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(conjugate_transpose(TU<M22> {3., 1, 0, 3}), ml));
  EXPECT_TRUE(is_near(conjugate_transpose(TL<CM22> {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}), TU<CM22> {cdouble(3,-1), cdouble(1,-2), 0, cdouble(3,1)}));
  EXPECT_TRUE(is_near(conjugate_transpose(TU<CM22> {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}), TL<CM22> {cdouble(3,-1), 0, cdouble(1,-2), cdouble(3,1)}));
  //
  EXPECT_NEAR(determinant(TL<M22> {3., 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(TU<M22> {3., 1, 0, 3}), 9, 1e-6);
  //
  EXPECT_NEAR(trace(TL<M22> {3., 0, 1, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(TU<M22> {3., 1, 0, 3}), 6, 1e-6);
  //
  EXPECT_TRUE(is_near(average_reduce<1>(TL<M22> {3., 0, 1, 3}), make_eigen_matrix(1.5, 2)));
  EXPECT_TRUE(is_near(average_reduce<1>(TU<M22> {3., 1, 0, 3}), make_eigen_matrix(2, 1.5)));
  //
  EXPECT_TRUE(is_near(average_reduce<0>(TL<M22> {3., 0, 1, 3}), make_eigen_matrix<double, 1, 2>(2, 1.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(TU<M22> {3., 1, 0, 3}), make_eigen_matrix<double, 1, 2>(1.5, 2)));
}


TEST(stl_interfaces, TriangularAdapter_contract)
{
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::TU<M22>>(), m33.template triangularView<Eigen::TU<M22>>()),
    make_dense_object_from<M33>(1, 12, 42, 0, 25, 84, 0, 0, 81)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::TU<M22>>(),
    m33.template triangularView<Eigen::TU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::TL<M22>>(), m33.template triangularView<Eigen::TL<M22>>()),
    make_dense_object_from<M33>(1, 0, 0, 24, 25, 0, 102, 112, 81)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::TL<M22>>(),
    m33.template triangularView<Eigen::TL<M22>>())), triangle_type::lower>);

  EXPECT_TRUE(is_near(contract(m3x_3.template triangularView<Eigen::TU<M22>>(), mx3_3.template triangularView<Eigen::TU<M22>>()),
    make_dense_object_from<M33>(1, 12, 42, 0, 25, 84, 0, 0, 81)));
  static_assert(triangular_matrix<decltype(contract(m3x_3.template triangularView<Eigen::TU<M22>>(),
    mx3_3.template triangularView<Eigen::TU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(contract(m3x_3.template triangularView<Eigen::TL<M22>>(), mx3_3.template triangularView<Eigen::TL<M22>>()),
    make_dense_object_from<M33>(1, 0, 0, 24, 25, 0, 102, 112, 81)));
  static_assert(triangular_matrix<decltype(contract(m3x_3.template triangularView<Eigen::TL<M22>>(),
    mx3_3.template triangularView<Eigen::TL<M22>>())), triangle_type::lower>);
}


TEST(stl_interfaces, TriangularAdapter_solve)
{
  EXPECT_TRUE(is_near(solve(TL<M22> {3., 0, 1, 3}, make_eigen_matrix<double, 2, 1>(3, 7)), make_eigen_matrix(1., 2)));
  EXPECT_TRUE(is_near(solve(TU<M22> {3., 1, 0, 3}, make_eigen_matrix<double, 2, 1>(3, 9)), make_eigen_matrix(0., 3)));
  //
  EXPECT_TRUE(is_near(solve(TL<M22> {3., 0, 1, 3}, make_eigen_matrix<double, 2, 1>(3, 7)), make_eigen_matrix<double, 2, 1>(1, 2)));
  EXPECT_TRUE(is_near(solve(TU<M22> {3., 1, 0, 3}, make_eigen_matrix<double, 2, 1>(3, 9)), make_eigen_matrix<double, 2, 1>(0, 3)));

  auto m22_3104 = make_dense_object_from<M22>(3, 1, 0, 4);
  auto m2x_3104 = M2x {m22_3104};
  auto mx2_3104 = Mx2 {m22_3104};
  auto mxx_3104 = Mxx {m22_3104};

  auto m22_5206 = make_dense_object_from<M22>(5, 2, 0, 6);

  auto m22_1512024 = make_eigen_matrix<double, 2, 2>(15, 12, 0, 24);
  auto m2x_1512024 = M2x {m22_1512024};
  auto mx2_1512024 = Mx2 {m22_1512024};
  auto mxx_1512024 = Mxx {m22_1512024};

  static_assert(triangular_matrix<decltype(solve(m22_3104.template triangularView<Eigen::TU<M22>>(), m22_1512024.template triangularView<Eigen::TU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::TU<M22>>(), m22_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::TU<M22>>(), m2x_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::TU<M22>>(), mx2_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::TU<M22>>(), mxx_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));

  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::TU<M22>>(), m22_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::TU<M22>>(), m2x_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::TU<M22>>(), mx2_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::TU<M22>>(), mxx_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::TU<M22>>(), m22_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::TU<M22>>(), m2x_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::TU<M22>>(), mx2_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  static_assert(triangular_matrix<decltype(solve(mx2_3104.template triangularView<Eigen::TU<M22>>(), mx2_1512024.template triangularView<Eigen::TU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::TU<M22>>(), mxx_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::TU<M22>>(), m22_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::TU<M22>>(), m2x_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::TU<M22>>(), mx2_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::TU<M22>>(), mxx_1512024.template triangularView<Eigen::TU<M22>>()), m22_5206));
}


TEST(stl_interfaces, TriangularAdapter_decompositions)
{
  EXPECT_TRUE(is_near(LQ_decomposition(TL<M22> {3., 0, 1, 3}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(QR_decomposition(TU<M22> {3., 1, 0, 3}), mat22(3., 1, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(TU<M22> {3., 1, 0, 3})), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(TL<M22> {3., 0, 1, 3})), mat22(10, 3, 3, 9)));
}


TEST(stl_interfaces, TriangularAdapter_blocks_lower)
{
  auto ma = triangular_adapter<eigen_matrix_t<double, 3, 3>, triangle_type::lower> {1, 0, 0,
                                                                                     2, 4, 0,
                                                                                     3, 5, 6};
  auto mb = triangular_adapter<eigen_matrix_t<double, 3, 3>, triangle_type::lower> {4, 0, 0,
                                                                                     5, 7, 0,
                                                                                     6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 0, 2, 3}, mb),
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}));
  static_assert(triangular_matrix<decltype(concatenate_diagonal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 0, 2, 3}, mb)), triangle_type::lower>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 0, 0,
                                    2, 4, 0,
                                    3, 5, 6,
                                    4, 0, 0,
                                    5, 7, 0,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3, 6>(1., 0, 0, 4, 0, 0,
                                     2, 4, 0, 5, 7, 0,
                                     3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 0, 2, 3}, mb}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 0, 2, 3},
               triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {4., 0, 5, 7}}));
  EXPECT_TRUE(is_near(split_vertical(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 0, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_eigen_matrix<3,5>(0., 0, 4, 0, 0,
                                               0, 0, 5, 7, 0,
                                               0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 0, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_eigen_matrix<2,5>(0., 0, 4, 0, 0,
                                               0, 0, 5, 7, 0)}));
  EXPECT_TRUE(is_near(split_horizontal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1, 0, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 0, 0,
                                                                              0, 0, 5, 7, 0,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 0, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 0, 0, 5, 7, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 0, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,2>(0., 0, 0, 0, 4, 0, 5, 7, 6, 8)}));

  EXPECT_TRUE(is_near(column(mb, 2), make_eigen_matrix(0., 0, 9)));
  EXPECT_TRUE(is_near(column<1>(mb), make_eigen_matrix(0., 7, 8)));
  EXPECT_TRUE(is_near(row(mb, 2), make_eigen_matrix<double, 1, 3>(6., 8, 9)));
  EXPECT_TRUE(is_near(row<1>(mb), make_eigen_matrix<double, 1, 3>(5., 7, 0)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 1, 1,
      6, 8, 1,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 1, 2,
      5, 8, 2,
      6, 9, 11)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 1, 1,
      6, 8, 1,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 0, 0,
      6, 8, 1,
      8, 10, 11)));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 1, 1,
      6, 8, 1,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 1, 2,
      6, 9, 3,
      8, 11, 13)));
}


TEST(stl_interfaces, TriangularAdapter_blocks_upper)
{
  auto ma = triangular_adapter<eigen_matrix_t<double, 3, 3>, triangle_type::upper> {1, 2, 3,
                                                                                     0, 4, 5,
                                                                                     0, 0, 6};
  auto mb = triangular_adapter<eigen_matrix_t<double, 3, 3>, triangle_type::upper> {4., 5, 6,
                                                                                     0, 7, 8,
                                                                                     0, 0, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 0, 3}, mb),
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}));
  static_assert(triangular_matrix<decltype(concatenate_diagonal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 0, 3}, mb)), triangle_type::upper>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 2, 3,
                                    0, 4, 5,
                                    0, 0, 6,
                                    4, 5, 6,
                                    0, 7, 8,
                                    0, 0, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    0, 4, 5, 0, 7, 8,
                                    0, 0, 6, 0, 0, 9)));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 0, 3}, mb}));
  EXPECT_TRUE(is_near(split_diagonal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 0, 3},
               triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {4., 5, 0, 7}}));
  EXPECT_TRUE(is_near(split_vertical(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                               0, 3, 0, 0, 0),
               make_eigen_matrix<3,5>(0., 0, 4, 5, 6,
                                               0, 0, 0, 7, 8,
                                               0, 0, 0, 0, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                               0, 3, 0, 0, 0),
               make_eigen_matrix<2,5>(0., 0, 4, 5, 6,
                                               0, 0, 0, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(triangular_adapter<eigen_matrix_t<double, 2, 2>, triangle_type::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 0, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 0, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    triangular_adapter<eigen_matrix_t<double, 5, 5>, triangle_type::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 0, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,2>(0., 0, 0, 0, 4, 5, 0, 7, 0, 0)}));

  EXPECT_TRUE(is_near(column(mb, 2), make_eigen_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(mb), make_eigen_matrix(5., 7, 0)));

  EXPECT_TRUE(is_near(row(mb, 2), make_eigen_matrix<double, 1, 3>(0., 0, 9)));
  EXPECT_TRUE(is_near(row<1>(mb), make_eigen_matrix<double, 1, 3>(0., 7, 8)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      1, 8, 9,
      1, 1, 10)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      0, 8, 10,
      0, 1, 11)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      1, 8, 9,
      1, 1, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      1, 9, 11,
      2, 3, 13)));
}


TEST(stl_interfaces, TriangularAdapter_arithmetic_lower)
{
  auto ma = TL<M22> {4., 0, 5, 6};
  auto mb = TL<M22> {1., 0, 2, 3};
  auto d = to_diagonal_adapter<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M22::Identity();
  auto z = zero_adapter<eigen_matrix_t<double, 2, 2>> {};

  EXPECT_TRUE(is_near(ma + mb, mat22(5, 0, 7, 9))); static_assert(triangular_matrix<decltype(ma + mb), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma + d, mat22(5, 0, 5, 9))); static_assert(triangular_matrix<decltype(ma + d), triangle_type::lower>);
  EXPECT_TRUE(is_near(d + ma, mat22(5, 0, 5, 9))); static_assert(triangular_matrix<decltype(d + ma), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma + i, mat22(5, 0, 5, 7))); static_assert(triangular_matrix<decltype(ma + i), triangle_type::lower>);
  EXPECT_TRUE(is_near(i + ma, mat22(5, 0, 5, 7))); static_assert(triangular_matrix<decltype(i + ma), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma + z, mat22(4, 0, 5, 6))); static_assert(triangular_matrix<decltype(ma + z), triangle_type::lower>);
  EXPECT_TRUE(is_near(z + ma, mat22(4, 0, 5, 6))); static_assert(triangular_matrix<decltype(z + ma), triangle_type::lower>);

  EXPECT_TRUE(is_near(ma - mb, mat22(3, 0, 3, 3))); static_assert(triangular_matrix<decltype(ma - mb), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma - d, mat22(3, 0, 5, 3))); static_assert(triangular_matrix<decltype(ma - d), triangle_type::lower>);
  EXPECT_TRUE(is_near(d - ma, mat22(-3, 0, -5, -3))); static_assert(triangular_matrix<decltype(d - ma), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma - i, mat22(3, 0, 5, 5))); static_assert(triangular_matrix<decltype(ma - i), triangle_type::lower>);
  EXPECT_TRUE(is_near(i - ma, mat22(-3, 0, -5, -5))); static_assert(triangular_matrix<decltype(i - ma), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma - z, mat22(4, 0, 5, 6))); static_assert(triangular_matrix<decltype(ma - z), triangle_type::lower>);
  EXPECT_TRUE(is_near(z - ma, mat22(-4, 0, -5, -6))); static_assert(triangular_matrix<decltype(z - ma), triangle_type::lower>);

  EXPECT_TRUE(is_near(ma * 2, mat22(8, 0, 10, 12))); static_assert(triangular_matrix<decltype(ma * 2), triangle_type::lower>);
  EXPECT_TRUE(is_near(2 * ma, mat22(8, 0, 10, 12))); static_assert(triangular_matrix<decltype(2 * ma), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma / 2, mat22(2, 0, 2.5, 3))); static_assert(triangular_matrix<decltype(ma / 2), triangle_type::lower>);
  static_assert(triangular_matrix<decltype(ma / 0), triangle_type::lower>);
  EXPECT_TRUE(is_near(-ma, mat22(-4, 0, -5, -6)));  static_assert(triangular_matrix<decltype(-ma), triangle_type::lower>);

  EXPECT_TRUE(is_near(triangular_adapter<decltype(i), triangle_type::diagonal> {i} * 2, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(triangular_adapter<decltype(i), triangle_type::diagonal> {i} * 2)>);
  EXPECT_TRUE(is_near(2 * triangular_adapter<decltype(i), triangle_type::diagonal> {i}, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(2 * triangular_adapter<decltype(i), triangle_type::diagonal> {i})>);
  EXPECT_TRUE(is_near(triangular_adapter<decltype(i), triangle_type::diagonal> {i} / 0.5, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(triangular_adapter<decltype(i), triangle_type::diagonal> {i} / 2)>);

  EXPECT_TRUE(is_near(ma * mb, mat22(4, 0, 17, 18))); static_assert(triangular_matrix<decltype(ma * mb), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma * d, mat22(4, 0, 5, 18))); static_assert(triangular_matrix<decltype(ma * d), triangle_type::lower>);
  EXPECT_TRUE(is_near(d * ma, mat22(4, 0, 15, 18))); static_assert(triangular_matrix<decltype(d * mb), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma * i, ma));  static_assert(triangular_matrix<decltype(ma * i), triangle_type::lower>);
  EXPECT_TRUE(is_near(i * ma, ma));  static_assert(triangular_matrix<decltype(i * ma), triangle_type::lower>);
  EXPECT_TRUE(is_near(ma * z, z));  static_assert(zero_matrix<decltype(ma * z)>);
  EXPECT_TRUE(is_near(z * ma, z));  static_assert(zero_matrix<decltype(z * ma)>);

  EXPECT_TRUE(is_near(mat22(4, 0, 5, 6) * mb, mat22(4, 0, 17, 18)));
  EXPECT_TRUE(is_near(ma * mat22(1, 0, 2, 3), mat22(4, 0, 17, 18)));
}


TEST(stl_interfaces, TriangularAdapter_arithmetic_upper)
{
  auto ma = TU<M22> {4., 5, 0, 6};
  auto mb = TU<M22> {1., 2, 0, 3};
  auto d = to_diagonal_adapter<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M22::Identity();
  auto z = zero_adapter<eigen_matrix_t<double, 2, 2>> {};
  EXPECT_TRUE(is_near(ma + mb, mat22(5, 7, 0, 9))); static_assert(triangular_matrix<decltype(ma + mb), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma + d, mat22(5, 5, 0, 9))); static_assert(triangular_matrix<decltype(ma + d), triangle_type::upper>);
  EXPECT_TRUE(is_near(d + ma, mat22(5, 5, 0, 9))); static_assert(triangular_matrix<decltype(d + ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma + i, mat22(5, 5, 0, 7))); static_assert(triangular_matrix<decltype(ma + i), triangle_type::upper>);
  EXPECT_TRUE(is_near(i + ma, mat22(5, 5, 0, 7))); static_assert(triangular_matrix<decltype(i + ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma + z, mat22(4, 5, 0, 6))); static_assert(triangular_matrix<decltype(ma + z), triangle_type::upper>);
  EXPECT_TRUE(is_near(z + ma, mat22(4, 5, 0, 6))); static_assert(triangular_matrix<decltype(z + ma), triangle_type::upper>);

  EXPECT_TRUE(is_near(ma - mb, mat22(3, 3, 0, 3))); static_assert(triangular_matrix<decltype(ma - mb), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma - d, mat22(3, 5, 0, 3))); static_assert(triangular_matrix<decltype(ma - d), triangle_type::upper>);
  EXPECT_TRUE(is_near(d - ma, mat22(-3, -5, 0, -3))); static_assert(triangular_matrix<decltype(d - ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma - i, mat22(3, 5, 0, 5))); static_assert(triangular_matrix<decltype(ma - i), triangle_type::upper>);
  EXPECT_TRUE(is_near(i - ma, mat22(-3, -5, 0, -5))); static_assert(triangular_matrix<decltype(i - ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma - z, mat22(4, 5, 0, 6))); static_assert(triangular_matrix<decltype(ma - z), triangle_type::upper>);
  EXPECT_TRUE(is_near(z - ma, mat22(-4, -5, 0, -6))); static_assert(triangular_matrix<decltype(z - ma), triangle_type::upper>);

  EXPECT_TRUE(is_near(ma * 2, mat22(8, 10, 0, 12))); static_assert(triangular_matrix<decltype(ma * 2), triangle_type::upper>);
  EXPECT_TRUE(is_near(2 * ma, mat22(8, 10, 0, 12))); static_assert(triangular_matrix<decltype(2 * ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma / 2, mat22(2, 2.5, 0, 3))); static_assert(triangular_matrix<decltype(ma / 2), triangle_type::upper>);
  static_assert(triangular_matrix<decltype(ma / 0), triangle_type::upper>);
  EXPECT_TRUE(is_near(-ma, mat22(-4, -5, 0, -6)));  static_assert(triangular_matrix<decltype(-ma), triangle_type::upper>);

  EXPECT_TRUE(is_near(ma * mb, mat22(4, 23, 0, 18))); static_assert(triangular_matrix<decltype(ma * mb), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma * d, mat22(4, 15, 0, 18))); static_assert(triangular_matrix<decltype(ma * d), triangle_type::upper>);
  EXPECT_TRUE(is_near(d * ma, mat22(4, 5, 0, 18))); static_assert(triangular_matrix<decltype(d * ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma * i, ma));  static_assert(triangular_matrix<decltype(ma * i), triangle_type::upper>);
  EXPECT_TRUE(is_near(i * ma, ma));  static_assert(triangular_matrix<decltype(i * ma), triangle_type::upper>);
  EXPECT_TRUE(is_near(ma * z, z));  static_assert(zero_matrix<decltype(ma * z)>);
  EXPECT_TRUE(is_near(z * ma, z));  static_assert(zero_matrix<decltype(z * ma)>);

  EXPECT_TRUE(is_near(mat22(4, 5, 0, 6) * mb, mat22(4, 23, 0, 18)));
  EXPECT_TRUE(is_near(ma * mat22(1, 2, 0, 3), mat22(4, 23, 0, 18)));
}


TEST(stl_interfaces, TriangularAdapter_arithmetic_mixed)
{
  auto m_upper = TU<M22> {4., 5, 0, 6};
  auto m_lower = TL<M22> {1., 0, 2, 3};
  EXPECT_TRUE(is_near(m_upper + m_lower, mat22(5, 5, 2, 9)));
  EXPECT_TRUE(is_near(m_lower + m_upper, mat22(5, 5, 2, 9)));
  EXPECT_TRUE(is_near(m_upper - m_lower, mat22(3, 5, -2, 3)));
  EXPECT_TRUE(is_near(m_lower - m_upper, mat22(-3, -5, 2, -3)));
  EXPECT_TRUE(is_near(m_upper * m_lower, mat22(14, 15, 12, 18)));
  EXPECT_TRUE(is_near(m_lower * m_upper, mat22(4, 5, 8, 28)));
}


TEST(stl_interfaces, TriangularAdapter_references)
{
  M22 m, n;
  m << 2, 0, 1, 2;
  n << 3, 0, 1, 3;
  using TL = triangular_adapter<M22, triangle_type::lower>;
  TL x {m};
  triangular_adapter<M22&, triangle_type::lower> x_lvalue {x};
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = TL {n};
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = TL {m};
  EXPECT_TRUE(is_near(x, m));
  EXPECT_TRUE(is_near(triangular_adapter<M22&, triangle_type::lower> {m}.nested_matrix(), mat22(2, 0, 1, 2)));
  //
  using V = triangular_adapter<eigen_matrix_t<double, 3, 3>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  bool test = false;
  try { v1(0, 1) = 3.2; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  v1(0, 1) = 0;
  EXPECT_EQ(v1(1,0), 2);
  triangular_adapter<eigen_matrix_t<double, 3, 3>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  triangular_adapter<eigen_matrix_t<double, 3, 3>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  triangular_adapter<const eigen_matrix_t<double, 3, 3>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  triangular_adapter<eigen_matrix_t<double, 3, 3>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}
*/