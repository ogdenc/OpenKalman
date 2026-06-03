/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests relating to hermitian_adapter.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"

#include "linear-algebra/adapters/triangular_adapter.hpp"
#include "linear-algebra/adapters/hermitian_adapter.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;
using namespace OpenKalman::patterns;


namespace
{
  using cdouble = std::complex<double>;

  using M1 = stdex::mdspan<double, stdex::extents<std::size_t, 1>>;
  using Mx = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent>>;
  using M11 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 1>>;
  using M1x = stdex::mdspan<double, stdex::extents<std::size_t, 1, stdex::dynamic_extent>>;
  using Mx1 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 1>>;
  using Mxx = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>>;

  using CM1 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 1>>;
  using CM11 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 1, 1>>;
  using CM1x = stdex::mdspan<cdouble, stdex::extents<std::size_t, 1, stdex::dynamic_extent>>;
  using CMx1 = stdex::mdspan<cdouble, stdex::extents<std::size_t, stdex::dynamic_extent, 1>>;
  using CMxx = stdex::mdspan<cdouble, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>>;

  using M2 = stdex::mdspan<double, stdex::extents<std::size_t, 2>>;
  using M21 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 1>>;
  using M12 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 2>>;
  using M22 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 2>>;
  using M2x = stdex::mdspan<double, stdex::extents<std::size_t, 2, stdex::dynamic_extent>>;
  using Mx2 = stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>;

  using CM2 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 2>>;
  using CM21 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 2, 1>>;
  using CM12 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 1, 2>>;
  using CM22 = stdex::mdspan<cdouble, stdex::extents<std::size_t, 2, 2>>;
  using CM2x = stdex::mdspan<cdouble, stdex::extents<std::size_t, 2, stdex::dynamic_extent>>;
  using CMx2 = stdex::mdspan<cdouble, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>;

  template<typename T> using HL = hermitian_adapter<T, triangle_type::lower>;
  template<typename T> using HU = hermitian_adapter<T, triangle_type::upper>;

  template<typename T> using TL = triangular_adapter<T, triangle_type::lower>;
  template<typename T> using TU = triangular_adapter<T, triangle_type::upper>;
  template<typename T> using TD = triangular_adapter<T, triangle_type::diagonal>;

  template<typename N, triangle_type tri>
  using MH = stdex::mdspan<
    typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::element_type,
    typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::extents_type,
    interface::layout_triangle_partition<typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::layout_type, tri>,
    interface::hermitian_accessor<typename std::decay_t<decltype(get_mdspan(std::declval<N>()))>::accessor_type>>;

  template<typename N>
  using MHL = MH<N, triangle_type::lower>;

  template<typename N>
  using MHU = MH<N, triangle_type::upper>;

}

#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"

TEST(stl_interfaces, hermitian_adapter_static_properties)
{
  static_assert(square_shaped<HL<M1>>);
  static_assert(square_shaped<HU<M1>>);
  static_assert(square_shaped<HL<M11>>);
  static_assert(square_shaped<HU<M11>>);

  static_assert(square_shaped<MHL<M1>>);
  static_assert(square_shaped<MHU<M1>>);
  static_assert(square_shaped<MHL<M11>>);
  static_assert(square_shaped<MHU<M11>>);

  static_assert(square_shaped<HL<Mx>>);
  static_assert(square_shaped<HU<Mx>>);
  static_assert(square_shaped<HL<Mxx>>);
  static_assert(square_shaped<HU<Mxx>>);

  static_assert(square_shaped<MHL<Mx>>);
  static_assert(square_shaped<MHU<Mx>>);
  static_assert(square_shaped<MHL<Mxx>>);
  static_assert(square_shaped<MHU<Mxx>>);

  static_assert(square_shaped<HL<Mx1>>);
  static_assert(square_shaped<HU<Mx1>>);
  static_assert(square_shaped<HL<M1x>>);
  static_assert(square_shaped<HU<M1x>>);

  static_assert(square_shaped<MHL<Mx1>>);
  static_assert(square_shaped<MHU<Mx1>>);
  static_assert(square_shaped<MHL<M1x>>);
  static_assert(square_shaped<MHU<M1x>>);

  static_assert(square_shaped<HL<M22>>);
  static_assert(square_shaped<HU<M22>>);
  static_assert(square_shaped<HL<M2x>>);
  static_assert(square_shaped<HU<M2x>>);
  static_assert(square_shaped<HL<Mx2>>);
  static_assert(square_shaped<HU<Mx2>>);
  static_assert(square_shaped<HL<Mxx>>);
  static_assert(square_shaped<HU<Mxx>>);
  static_assert(square_shaped<HL<CM22>>);
  static_assert(square_shaped<HU<CM22>>);

  static_assert(square_shaped<MHL<M22>>);
  static_assert(square_shaped<MHU<M22>>);
  static_assert(square_shaped<MHL<M2x>>);
  static_assert(square_shaped<MHU<M2x>>);
  static_assert(square_shaped<MHL<Mx2>>);
  static_assert(square_shaped<MHU<Mx2>>);
  static_assert(square_shaped<MHL<Mxx>>);
  static_assert(square_shaped<MHU<Mxx>>);
  static_assert(square_shaped<MHL<CM22>>);
  static_assert(square_shaped<MHU<CM22>>);

  static_assert(one_dimensional<HL<M1>>);
  static_assert(one_dimensional<HU<M1>>);
  static_assert(one_dimensional<HL<M11>>);
  static_assert(one_dimensional<HU<M11>>);

  static_assert(one_dimensional<MHL<M1>>);
  static_assert(one_dimensional<MHU<M1>>);
  static_assert(one_dimensional<MHL<M11>>);
  static_assert(one_dimensional<MHU<M11>>);

  static_assert(one_dimensional<HL<Mx>>);
  static_assert(one_dimensional<HU<Mx>>);
  static_assert(not one_dimensional<HL<Mxx>>);
  static_assert(not one_dimensional<HU<Mxx>>);

  static_assert(one_dimensional<MHL<Mx>>);
  static_assert(one_dimensional<MHU<Mx>>);
  static_assert(not one_dimensional<MHL<Mxx>>);
  static_assert(not one_dimensional<MHU<Mxx>>);

  static_assert(one_dimensional<HL<Mx1>>);
  static_assert(one_dimensional<HU<Mx1>>);
  static_assert(one_dimensional<HL<M1x>>);
  static_assert(one_dimensional<HU<M1x>>);

  static_assert(one_dimensional<MHL<Mx1>>);
  static_assert(one_dimensional<MHU<Mx1>>);
  static_assert(one_dimensional<MHL<M1x>>);
  static_assert(one_dimensional<MHU<M1x>>);

  static_assert(diagonal_matrix<HL<M1>>);
  static_assert(diagonal_matrix<HU<M1>>);
  static_assert(diagonal_matrix<HL<M11>>);
  static_assert(diagonal_matrix<HU<M11>>);

  static_assert(diagonal_matrix<MHL<M1>>);
  static_assert(diagonal_matrix<MHU<M1>>);
  static_assert(diagonal_matrix<MHL<M11>>);
  static_assert(diagonal_matrix<MHU<M11>>);

  static_assert(diagonal_matrix<HL<Mx>>);
  static_assert(diagonal_matrix<HU<Mx>>);
  static_assert(not diagonal_matrix<HL<Mxx>>);
  static_assert(not diagonal_matrix<HU<Mxx>>);

  static_assert(diagonal_matrix<MHL<Mx>>);
  static_assert(diagonal_matrix<MHU<Mx>>);
  static_assert(not diagonal_matrix<MHL<Mxx>>);
  static_assert(not diagonal_matrix<MHU<Mxx>>);

  static_assert(diagonal_matrix<HL<Mx1>>);
  static_assert(diagonal_matrix<HU<Mx1>>);
  static_assert(diagonal_matrix<HL<M1x>>);
  static_assert(diagonal_matrix<HU<M1x>>);

  static_assert(diagonal_matrix<MHL<Mx1>>);
  static_assert(diagonal_matrix<MHU<Mx1>>);
  static_assert(diagonal_matrix<MHL<M1x>>);
  static_assert(diagonal_matrix<MHU<M1x>>);

  static_assert(diagonal_matrix<HU<TL<M22>>>);
  static_assert(diagonal_matrix<HL<TU<M22>>>);
  static_assert(diagonal_matrix<HL<TD<M22>>>);
  static_assert(diagonal_matrix<HU<TD<M22>>>);
  static_assert(diagonal_matrix<TD<HL<M22>>>);
  static_assert(diagonal_matrix<TD<HU<M22>>>);

  static_assert(diagonal_matrix<MHU<TL<M22>>>);
  static_assert(diagonal_matrix<MHL<TU<M22>>>);
  static_assert(diagonal_matrix<MHL<TD<M22>>>);
  static_assert(diagonal_matrix<MHU<TD<M22>>>);
  static_assert(diagonal_matrix<TD<MHL<M22>>>);
  static_assert(diagonal_matrix<TD<MHU<M22>>>);

  static_assert(diagonal_matrix<HU<TL<CM22>>>);
  static_assert(diagonal_matrix<HL<TU<CM22>>>);
  static_assert(diagonal_matrix<HL<TD<CM22>>>);
  static_assert(diagonal_matrix<HU<TD<CM22>>>);
  static_assert(diagonal_matrix<TD<HL<CM22>>>);
  static_assert(diagonal_matrix<TD<HU<CM22>>>);

  static_assert(diagonal_matrix<MHU<TL<CM22>>>);
  static_assert(diagonal_matrix<MHL<TU<CM22>>>);
  static_assert(diagonal_matrix<MHL<TD<CM22>>>);
  static_assert(diagonal_matrix<MHU<TD<CM22>>>);
  static_assert(diagonal_matrix<TD<MHL<CM22>>>);
  static_assert(diagonal_matrix<TD<MHU<CM22>>>);

  static_assert(not diagonal_matrix<HL<M22>>);
  static_assert(not diagonal_matrix<HU<M22>>);
  static_assert(not diagonal_matrix<HL<M2x>>);
  static_assert(not diagonal_matrix<HU<M2x>>);
  static_assert(not diagonal_matrix<HL<Mx2>>);
  static_assert(not diagonal_matrix<HU<Mx2>>);
  static_assert(not diagonal_matrix<HL<Mxx>>);
  static_assert(not diagonal_matrix<HU<Mxx>>);
  static_assert(not diagonal_matrix<HL<CM22>>);
  static_assert(not diagonal_matrix<HU<CM22>>);

  static_assert(not diagonal_matrix<MHL<M22>>);
  static_assert(not diagonal_matrix<MHU<M22>>);
  static_assert(not diagonal_matrix<MHL<M2x>>);
  static_assert(not diagonal_matrix<MHU<M2x>>);
  static_assert(not diagonal_matrix<MHL<Mx2>>);
  static_assert(not diagonal_matrix<MHU<Mx2>>);
  static_assert(not diagonal_matrix<MHL<Mxx>>);
  static_assert(not diagonal_matrix<MHU<Mxx>>);
  static_assert(not diagonal_matrix<MHL<CM22>>);
  static_assert(not diagonal_matrix<MHU<CM22>>);

  static_assert(hermitian_matrix<HL<Mx>>);
  static_assert(hermitian_matrix<HU<Mx>>);
  static_assert(hermitian_matrix<HL<Mxx>>);
  static_assert(hermitian_matrix<HU<Mxx>>);

  static_assert(hermitian_matrix<MHL<Mx>>);
  static_assert(hermitian_matrix<MHU<Mx>>);
  static_assert(hermitian_matrix<MHL<Mxx>>);
  static_assert(hermitian_matrix<MHU<Mxx>>);

  static_assert(hermitian_matrix<HL<M22>>);
  static_assert(hermitian_matrix<HU<M22>>);
  static_assert(hermitian_matrix<HL<M2x>>);
  static_assert(hermitian_matrix<HU<M2x>>);
  static_assert(hermitian_matrix<HL<Mx2>>);
  static_assert(hermitian_matrix<HU<Mx2>>);

  static_assert(hermitian_matrix<MHL<M22>>);
  static_assert(hermitian_matrix<MHU<M22>>);
  static_assert(hermitian_matrix<MHL<M2x>>);
  static_assert(hermitian_matrix<MHU<M2x>>);
  static_assert(hermitian_matrix<MHL<Mx2>>);
  static_assert(hermitian_matrix<MHU<Mx2>>);

  static_assert(hermitian_matrix<HL<CM22>>);
  static_assert(hermitian_matrix<HU<CM22>>);
  static_assert(hermitian_matrix<HL<CM2x>>);
  static_assert(hermitian_matrix<HU<CM2x>>);
  static_assert(hermitian_matrix<HL<CMx2>>);
  static_assert(hermitian_matrix<HU<CMx2>>);
  static_assert(hermitian_matrix<HL<CMxx>>);
  static_assert(hermitian_matrix<HU<CMxx>>);

  static_assert(hermitian_matrix<MHL<CM22>>);
  static_assert(hermitian_matrix<MHU<CM22>>);
  static_assert(hermitian_matrix<MHL<CM2x>>);
  static_assert(hermitian_matrix<MHU<CM2x>>);
  static_assert(hermitian_matrix<MHL<CMx2>>);
  static_assert(hermitian_matrix<MHU<CMx2>>);
  static_assert(hermitian_matrix<MHL<CMxx>>);
  static_assert(hermitian_matrix<MHU<CMxx>>);
}

#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/functions/copy_from.hpp"
#include "linear-algebra/functions/make_constant.hpp"
#include "linear-algebra/functions/make_zero.hpp"
#include "linear-algebra/functions/to_hermitian.hpp"

TEST(stl_interfaces, hermitian_adapter_dynamic_properties)
{
  using A22 = double[2][2];
  A22 a22 {{1, 2}, {3, 4}};

  decltype(auto) la22 = to_hermitian<triangle_type::lower>(a22);
  static_assert(hermitian_matrix<decltype(la22)>);
  EXPECT_EQ((la22[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((la22[std::array{0U, 1U}]), 3.);
  EXPECT_EQ((la22[std::array{1U, 0U}]), 3.);
  EXPECT_EQ((la22[std::array{1U, 1U}]), 4.);

  decltype(auto) ua22 = to_hermitian<triangle_type::upper>(a22);
  static_assert(hermitian_matrix<decltype(ua22)>);
  EXPECT_EQ((ua22[std::array{0U, 0U}]), 1.);
  EXPECT_EQ((ua22[std::array{0U, 1U}]), 2.);
  EXPECT_EQ((ua22[std::array{1U, 0U}]), 2.);
  EXPECT_EQ((ua22[std::array{1U, 1U}]), 4.);

  auto pa22 = attach_patterns(a22, std::tuple{Polar{}, Dimensions<2>{}});
  decltype(auto) pla22 = to_hermitian<triangle_type::lower>(pa22);
  static_assert(compares_with_pattern_collection<decltype(pla22), std::tuple<Polar<>, Dimensions<2>>, &stdex::is_eq, applicability::guaranteed>);

  auto a4 = std::array {cdouble{1., 0.1}, cdouble{2., 0.2}, cdouble{3., -0.3}, cdouble{4., 0.4}};
  auto maxx_22 = stdex::mdspan<cdouble, stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>>(a4.data(), 2, 2);

  decltype(auto) laxx_22 = to_hermitian<triangle_type::lower>(maxx_22);
  static_assert(hermitian_matrix<decltype(laxx_22)>);
  EXPECT_EQ((laxx_22[std::array{0U, 0U}]), (cdouble{1., 0.}));
  EXPECT_EQ((laxx_22[std::array{0U, 1U}]), (cdouble{3., .3}));
  EXPECT_EQ((laxx_22[std::array{1U, 0U}]), (cdouble{3., -.3}));
  EXPECT_EQ((laxx_22[std::array{1U, 1U}]), (cdouble{4., 0.}));

  decltype(auto) uaxx_22 = to_hermitian<triangle_type::upper>(maxx_22);
  static_assert(hermitian_matrix<decltype(uaxx_22)>);
  EXPECT_EQ((uaxx_22[std::array{0U, 0U}]), (cdouble{1., 0.}));
  EXPECT_EQ((uaxx_22[std::array{0U, 1U}]), (cdouble{2., .2}));
  EXPECT_EQ((uaxx_22[std::array{1U, 0U}]), (cdouble{2., -.2}));
  EXPECT_EQ((uaxx_22[std::array{1U, 1U}]), (cdouble{4., 0.}));


  auto a4w = std::array<cdouble, 4> {9., 9., 9., 9.};
  auto m22w = stdex::mdspan<cdouble, stdex::extents<std::size_t, 2, 2>>(a4w.data());
  EXPECT_FALSE(is_near(m22w, laxx_22));
  copy_from(m22w, laxx_22);
  EXPECT_TRUE(is_near(m22w, laxx_22));
  copy_from(m22w, uaxx_22);
  EXPECT_TRUE(is_near(m22w, uaxx_22));
  copy_from(m22w, la22);
  EXPECT_TRUE(is_near(m22w, la22));
  copy_from(m22w, ua22);
  EXPECT_TRUE(is_near(m22w, ua22));

  auto lz22 = to_hermitian<triangle_type::lower>(make_zero<double>(stdex::extents<std::size_t, 2, 2>{}));
  static_assert(triangular_matrix<decltype(lz22), triangle_type::lower>);
  EXPECT_EQ(constant_value(lz22), 0);
  EXPECT_EQ((lz22[std::array{0U, 0U}]), 0.);
  EXPECT_EQ((lz22[std::array{0U, 1U}]), 0.);
  EXPECT_EQ((lz22[std::array{1U, 0U}]), 0.);
  EXPECT_EQ((lz22[std::array{1U, 1U}]), 0.);

  auto mc22_5 = make_constant(5., stdex::extents<std::size_t, 2, 2>{});

  auto lc22_5 = to_hermitian<triangle_type::lower>(mc22_5);
  static_assert(hermitian_matrix<decltype(lc22_5)>);
  EXPECT_EQ((lc22_5[std::array{0U, 0U}]), 5.);
  EXPECT_EQ((lc22_5[std::array{0U, 1U}]), 5.);
  EXPECT_EQ((lc22_5[std::array{1U, 0U}]), 5.);
  EXPECT_EQ((lc22_5[std::array{1U, 1U}]), 5.);

  auto uc22_5 = to_hermitian<triangle_type::upper>(mc22_5);
  static_assert(hermitian_matrix<decltype(uc22_5)>);
  EXPECT_EQ((uc22_5[std::array{0U, 0U}]), 5.);
  EXPECT_EQ((uc22_5[std::array{0U, 1U}]), 5.);
  EXPECT_EQ((uc22_5[std::array{1U, 0U}]), 5.);
  EXPECT_EQ((uc22_5[std::array{1U, 1U}]), 5.);

  auto mc22c_5 = make_constant(cdouble{5., 0.5}, stdex::extents<std::size_t, 2, 2>{});

  auto lc22c_5 = to_hermitian<triangle_type::lower>(mc22c_5);
  static_assert(hermitian_matrix<decltype(lc22c_5)>);
  EXPECT_EQ((lc22c_5[std::array{0U, 0U}]), (cdouble{5., 0.}));
  EXPECT_EQ((lc22c_5[std::array{0U, 1U}]), (cdouble{5., -0.5}));
  EXPECT_EQ((lc22c_5[std::array{1U, 0U}]), (cdouble{5., 0.5}));
  EXPECT_EQ((lc22c_5[std::array{1U, 1U}]), (cdouble{5., 0.}));

  auto uc22c_5 = to_hermitian<triangle_type::upper>(mc22c_5);
  static_assert(hermitian_matrix<decltype(uc22c_5)>);
  EXPECT_EQ((uc22c_5[std::array{0U, 0U}]), (cdouble{5., 0.}));
  EXPECT_EQ((uc22c_5[std::array{0U, 1U}]), (cdouble{5., 0.5}));
  EXPECT_EQ((uc22c_5[std::array{1U, 0U}]), (cdouble{5., -0.5}));
  EXPECT_EQ((uc22c_5[std::array{1U, 1U}]), (cdouble{5., 0.}));
}


/*TEST(stl_interfaces, triangular_adapter_functions)
{
  auto a6 = std::array {9., 9., 9., 9., 9., 9.};
  auto m23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>(a6.data());
  decltype(auto) la23 = to_triangular<triangle_type::lower>(m23);

}*/

/*
TEST(stl_interfaces, TriangularAdapter_overloads)
{
  M22 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  //
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(HL<M22>(3., 0, 1, 3)), ml));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(HU<M22>(3., 1, 0, 3)), mu));
  //
  EXPECT_TRUE(is_near(make_self_contained(HL<M22>(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_self_contained(HU<M22>(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(HL<M22> {9, 3, 3, 10} * 2))>, HL<M22>>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(HU<M22> {9, 3, 3, 10} * 2))>, HU<M22>>);
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
  HL<M22> lower {3., 0, 1, 3};
  EXPECT_TRUE(is_near(Cholesky_square(lower), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(HL<M22> {3., 0, 1, 3}), mat22(9., 3, 3, 10)));
  static_assert(hermitian_adapter_concept<decltype(Cholesky_square(HL<M22> {3, 0, 1, 3})), triangle_type::lower>);
  //
  HU<M22> upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(Cholesky_square(upper), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(HU<M22> {3., 1, 0, 3}), mat22(9., 3, 3, 10)));
  static_assert(hermitian_adapter_concept<decltype(Cholesky_square(HU<M22> {3, 1, 0, 3})), triangle_type::upper>);
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
  EXPECT_TRUE(is_near(diagonal_of(HL<M22> {3., 0, 1, 3}), make_eigen_matrix<double, 2, 1>(3, 3)));
  EXPECT_TRUE(is_near(diagonal_of(HU<M22> {3., 1, 0, 3}), make_eigen_matrix<double, 2, 1>(3, 3)));
  //
  EXPECT_TRUE(is_near(transpose(HL<M22> {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(transpose(HU<M22> {3., 1, 0, 3}), ml));
  EXPECT_TRUE(is_near(transpose(HL<CM22> {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}), HU<CM22> {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}));
  EXPECT_TRUE(is_near(transpose(HU<CM22> {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}), HL<CM22> {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}));
  //
  EXPECT_TRUE(is_near(conjugate_transpose(HL<M22> {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(conjugate_transpose(HU<M22> {3., 1, 0, 3}), ml));
  EXPECT_TRUE(is_near(conjugate_transpose(HL<CM22> {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}), HU<CM22> {cdouble(3,-1), cdouble(1,-2), 0, cdouble(3,1)}));
  EXPECT_TRUE(is_near(conjugate_transpose(HU<CM22> {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}), HL<CM22> {cdouble(3,-1), 0, cdouble(1,-2), cdouble(3,1)}));
  //
  EXPECT_NEAR(determinant(HL<M22> {3., 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(HU<M22> {3., 1, 0, 3}), 9, 1e-6);
  //
  EXPECT_NEAR(trace(HL<M22> {3., 0, 1, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(HU<M22> {3., 1, 0, 3}), 6, 1e-6);
  //
  EXPECT_TRUE(is_near(average_reduce<1>(HL<M22> {3., 0, 1, 3}), make_eigen_matrix(1.5, 2)));
  EXPECT_TRUE(is_near(average_reduce<1>(HU<M22> {3., 1, 0, 3}), make_eigen_matrix(2, 1.5)));
  //
  EXPECT_TRUE(is_near(average_reduce<0>(HL<M22> {3., 0, 1, 3}), make_eigen_matrix<double, 1, 2>(2, 1.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(HU<M22> {3., 1, 0, 3}), make_eigen_matrix<double, 1, 2>(1.5, 2)));
}


TEST(stl_interfaces, TriangularAdapter_contract)
{
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::HU<M22>>(), m33.template triangularView<Eigen::HU<M22>>()),
    make_dense_object_from<M33>(1, 12, 42, 0, 25, 84, 0, 0, 81)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::HU<M22>>(),
    m33.template triangularView<Eigen::HU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::HL<M22>>(), m33.template triangularView<Eigen::HL<M22>>()),
    make_dense_object_from<M33>(1, 0, 0, 24, 25, 0, 102, 112, 81)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::HL<M22>>(),
    m33.template triangularView<Eigen::HL<M22>>())), triangle_type::lower>);

  EXPECT_TRUE(is_near(contract(m3x_3.template triangularView<Eigen::HU<M22>>(), mx3_3.template triangularView<Eigen::HU<M22>>()),
    make_dense_object_from<M33>(1, 12, 42, 0, 25, 84, 0, 0, 81)));
  static_assert(triangular_matrix<decltype(contract(m3x_3.template triangularView<Eigen::HU<M22>>(),
    mx3_3.template triangularView<Eigen::HU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(contract(m3x_3.template triangularView<Eigen::HL<M22>>(), mx3_3.template triangularView<Eigen::HL<M22>>()),
    make_dense_object_from<M33>(1, 0, 0, 24, 25, 0, 102, 112, 81)));
  static_assert(triangular_matrix<decltype(contract(m3x_3.template triangularView<Eigen::HL<M22>>(),
    mx3_3.template triangularView<Eigen::HL<M22>>())), triangle_type::lower>);
}


TEST(stl_interfaces, TriangularAdapter_solve)
{
  EXPECT_TRUE(is_near(solve(HL<M22> {3., 0, 1, 3}, make_eigen_matrix<double, 2, 1>(3, 7)), make_eigen_matrix(1., 2)));
  EXPECT_TRUE(is_near(solve(HU<M22> {3., 1, 0, 3}, make_eigen_matrix<double, 2, 1>(3, 9)), make_eigen_matrix(0., 3)));
  //
  EXPECT_TRUE(is_near(solve(HL<M22> {3., 0, 1, 3}, make_eigen_matrix<double, 2, 1>(3, 7)), make_eigen_matrix<double, 2, 1>(1, 2)));
  EXPECT_TRUE(is_near(solve(HU<M22> {3., 1, 0, 3}, make_eigen_matrix<double, 2, 1>(3, 9)), make_eigen_matrix<double, 2, 1>(0, 3)));

  auto m22_3104 = make_dense_object_from<M22>(3, 1, 0, 4);
  auto m2x_3104 = M2x {m22_3104};
  auto mx2_3104 = Mx2 {m22_3104};
  auto mxx_3104 = Mxx {m22_3104};

  auto m22_5206 = make_dense_object_from<M22>(5, 2, 0, 6);

  auto m22_1512024 = make_eigen_matrix<double, 2, 2>(15, 12, 0, 24);
  auto m2x_1512024 = M2x {m22_1512024};
  auto mx2_1512024 = Mx2 {m22_1512024};
  auto mxx_1512024 = Mxx {m22_1512024};

  static_assert(triangular_matrix<decltype(solve(m22_3104.template triangularView<Eigen::HU<M22>>(), m22_1512024.template triangularView<Eigen::HU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::HU<M22>>(), m22_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::HU<M22>>(), m2x_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::HU<M22>>(), mx2_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::HU<M22>>(), mxx_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));

  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::HU<M22>>(), m22_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::HU<M22>>(), m2x_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::HU<M22>>(), mx2_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::HU<M22>>(), mxx_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::HU<M22>>(), m22_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::HU<M22>>(), m2x_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::HU<M22>>(), mx2_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  static_assert(triangular_matrix<decltype(solve(mx2_3104.template triangularView<Eigen::HU<M22>>(), mx2_1512024.template triangularView<Eigen::HU<M22>>())), triangle_type::upper>);
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::HU<M22>>(), mxx_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::HU<M22>>(), m22_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::HU<M22>>(), m2x_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::HU<M22>>(), mx2_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::HU<M22>>(), mxx_1512024.template triangularView<Eigen::HU<M22>>()), m22_5206));
}


TEST(stl_interfaces, TriangularAdapter_decompositions)
{
  EXPECT_TRUE(is_near(LQ_decomposition(HL<M22> {3., 0, 1, 3}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(QR_decomposition(HU<M22> {3., 1, 0, 3}), mat22(3., 1, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(HU<M22> {3., 1, 0, 3})), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(HL<M22> {3., 0, 1, 3})), mat22(10, 3, 3, 9)));
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
  auto ma = HL<M22> {4., 0, 5, 6};
  auto mb = HL<M22> {1., 0, 2, 3};
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
  auto ma = HU<M22> {4., 5, 0, 6};
  auto mb = HU<M22> {1., 2, 0, 3};
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
  auto m_upper = HU<M22> {4., 5, 0, 6};
  auto m_lower = HL<M22> {1., 0, 2, 3};
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
  using HL = triangular_adapter<M22, triangle_type::lower>;
  HL x {m};
  triangular_adapter<M22&, triangle_type::lower> x_lvalue {x};
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = HL {n};
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = HL {m};
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