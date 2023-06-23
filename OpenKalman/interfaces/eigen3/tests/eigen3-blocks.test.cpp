/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M14 = eigen_matrix_t<double, 1, 4>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M24 = eigen_matrix_t<double, 2, 4>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M42 = eigen_matrix_t<double, 4, 2>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M45 = eigen_matrix_t<double, 4, 5>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
  using M40 = eigen_matrix_t<double, 4, dynamic_size>;
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, element_access)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);

  EXPECT_NEAR(m22(0, 0), 1, 1e-6);
  EXPECT_NEAR(m22(0, 1), 2, 1e-6);

  auto d1 = make_eigen_matrix<double, 3, 1>(1, 2, 3);
  EXPECT_NEAR(d1(1), 2, 1e-6);
  d1(0) = 5;
  d1(2, 0) = 7;
  EXPECT_TRUE(is_near(d1, make_eigen_matrix<double, 3, 1>(5, 2, 7)));
}


TEST(eigen3, get_and_set_elements)
{
  auto m21 = make_dense_writable_matrix_from<M21>(1, 2);
  auto m12 = make_dense_writable_matrix_from<M12>(1, 2);
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);

  M22 el22 {m22}; // 1, 2, 3, 4
  M20 el20_2 {m22};
  M02 el02_2 {m22};
  M00 el00_22 {m22};

  EXPECT_NEAR(get_element(el22, 0, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el20_2, 0, 1), 2, 1e-8);
  EXPECT_NEAR(get_element(el02_2, 1, 0), 3, 1e-8);
  EXPECT_NEAR(get_element(el00_22, 1, 1), 4, 1e-8);

  set_element(el22, 5.5, 1, 0); EXPECT_NEAR(get_element(el22, 1, 0), 5.5, 1e-8);
  set_element(el20_2, 5.5, 1, 0); EXPECT_NEAR(get_element(el20_2, 1, 0), 5.5, 1e-8);
  set_element(el02_2, 5.5, 1, 0); EXPECT_NEAR(get_element(el02_2, 1, 0), 5.5, 1e-8);
  set_element(el00_22, 5.5, 1, 0); EXPECT_NEAR(get_element(el00_22, 1, 0), 5.5, 1e-8);

  M21 el21 {m21}; // 1, 2
  M20 el20_1 {m21};
  M01 el01_2 {m21};
  M00 el00_21 {m21};

  EXPECT_NEAR(get_element(el21, 0, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el20_1, 1, 0), 2, 1e-8);
  EXPECT_NEAR(get_element(el01_2, 0, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el00_21, 1, 0), 2, 1e-8);

  set_element(el21, 5.5, 1, 0); EXPECT_NEAR(get_element(el21, 1, 0), 5.5, 1e-8);
  set_element(el20_1, 5.5, 1, 0); EXPECT_NEAR(get_element(el20_1, 1, 0), 5.5, 1e-8);
  set_element(el01_2, 5.5, 1, 0); EXPECT_NEAR(get_element(el01_2, 1, 0), 5.5, 1e-8);
  set_element(el00_21, 5.5, 1, 0); EXPECT_NEAR(get_element(el00_21, 1, 0), 5.5, 1e-8);

  set_element(el21, 5.6, 1); EXPECT_NEAR(get_element(el21, 1), 5.6, 1e-8);
  set_element(el01_2, 5.6, 1); EXPECT_NEAR(get_element(el01_2, 1), 5.6, 1e-8);

  M12 el12 {m12}; // 1, 2
  M10 el10_2 {m12};
  M02 el02_1 {m12};
  M00 el00_12 {m12};

  EXPECT_NEAR(get_element(el12, 0, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el10_2, 0, 1), 2, 1e-8);
  EXPECT_NEAR(get_element(el02_1, 0, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el00_12, 0, 1), 2, 1e-8);

  set_element(el12, 5.5, 0, 1); EXPECT_NEAR(get_element(el12, 0, 1), 5.5, 1e-8);
  set_element(el10_2, 5.5, 0, 1); EXPECT_NEAR(get_element(el10_2, 0, 1), 5.5, 1e-8);
  set_element(el02_1, 5.5, 0, 1); EXPECT_NEAR(get_element(el02_1, 0, 1), 5.5, 1e-8);
  set_element(el00_12, 5.5, 0, 1); EXPECT_NEAR(get_element(el00_12, 0, 1), 5.5, 1e-8);

  set_element(el12, 5.6, 1); EXPECT_NEAR(get_element(el12, 1), 5.6, 1e-8);
  set_element(el10_2, 5.6, 1); EXPECT_NEAR(get_element(el10_2, 1), 5.6, 1e-8);

  Eigen::DiagonalMatrix<double, 2> dm2{m21};
  EXPECT_EQ(get_element(dm2, 0, 0), 1);
  EXPECT_EQ(get_element(dm2, 0, 1), 0);
  EXPECT_EQ(get_element(dm2, 1, 1), 2);
  EXPECT_EQ(get_element(dm2, 1, 0), 0);
  EXPECT_EQ(get_element(dm2, 0), 1);
  EXPECT_EQ(get_element(dm2, 1), 2);

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> dm0{m21};
  EXPECT_EQ(get_element(dm0, 0, 0), 1);
  EXPECT_EQ(get_element(dm0, 0, 1), 0);
  EXPECT_EQ(get_element(dm0, 1, 1), 2);
  EXPECT_EQ(get_element(dm0, 1, 0), 0);
  EXPECT_EQ(get_element(dm0, 0), 1);
  EXPECT_EQ(get_element(dm0, 1), 2);
}


TEST(eigen3, get_block)
{
  auto N0 = std::integral_constant<std::size_t, 0>{};
  auto N1 = std::integral_constant<std::size_t, 1>{};
  auto N2 = std::integral_constant<std::size_t, 2>{};
  auto N3 = std::integral_constant<std::size_t, 3>{};

  auto m34 = make_dense_writable_matrix_from<M34>(
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12);

  EXPECT_TRUE(is_near(get_block(m34, std::tuple{N0, N0}, std::tuple{N2, N2}), make_dense_writable_matrix_from<M22>(1, 2, 5, 6)));
  EXPECT_TRUE(is_near(get_block(M34{m34}, std::tuple{N1, N1}, std::tuple{2, N3}), make_dense_writable_matrix_from<M23>(6, 7, 8, 10, 11, 12)));
  EXPECT_TRUE(is_near(get_block(m34, std::tuple{N1, 1}, std::tuple{N2, 2}), make_dense_writable_matrix_from<M22>(6, 7, 10, 11)));
  EXPECT_TRUE(is_near(get_block(m34, std::tuple{0, 0}, std::tuple{2, N2}), make_dense_writable_matrix_from<M22>(1, 2, 5, 6)));
  EXPECT_TRUE(is_near(get_block(M30{m34}, std::tuple{0, N0}, std::tuple{2, 2}), make_dense_writable_matrix_from<M22>(1, 2, 5, 6)));
  EXPECT_TRUE(is_near(get_block(M00{m34}, std::tuple{0, 0}, std::tuple{2, 2}), make_dense_writable_matrix_from<M22>(1, 2, 5, 6)));
  EXPECT_TRUE(is_near(get_block(M00{m34}, std::tuple{N0, N0}, std::tuple{N2, N2}), make_dense_writable_matrix_from<M22>(1, 2, 5, 6)));

  EXPECT_TRUE(is_near(get_block<0, 1>(M00{m34}, std::tuple{N0, N0}, std::tuple{N2, N3}), make_dense_writable_matrix_from<M23>(1, 2, 3, 5, 6, 7)));
  EXPECT_TRUE(is_near(get_block<1, 0>(M00{m34}, std::tuple{N0, N0}, std::tuple{N3, N2}), make_dense_writable_matrix_from<M23>(1, 2, 3, 5, 6, 7)));

  EXPECT_TRUE(is_near(get_block<0>(m34, std::tuple{N0}, std::tuple{N2}), make_dense_writable_matrix_from<M24>(1, 2, 3, 4, 5, 6, 7, 8)));
  EXPECT_TRUE(is_near(get_block<0>(m34, std::tuple{N1}, std::tuple{2}), make_dense_writable_matrix_from<M24>(5, 6, 7, 8, 9, 10, 11, 12)));
  EXPECT_TRUE(is_near(get_block<0>(m34, std::tuple{1}, std::tuple{1}), make_dense_writable_matrix_from<M14>(5, 6, 7, 8)));
  EXPECT_TRUE(is_near(get_block<0>(M00{m34}, std::tuple{1}, std::tuple{N2}), make_dense_writable_matrix_from<M24>(5, 6, 7, 8, 9, 10, 11, 12)));

  EXPECT_TRUE(is_near(get_block<1>(m34, std::tuple{N0}, std::tuple{N2}), make_dense_writable_matrix_from<M32>(1, 2, 5, 6, 9, 10)));
  EXPECT_TRUE(is_near(get_block<1>(m34, std::tuple{N1}, std::tuple{2}), make_dense_writable_matrix_from<M32>(2, 3, 6, 7, 10, 11)));
  EXPECT_TRUE(is_near(get_block<1>(m34, std::tuple{1}, std::tuple{1}), make_dense_writable_matrix_from<M31>(2, 6, 10)));
  EXPECT_TRUE(is_near(get_block<1>(M00{m34}, std::tuple{1}, std::tuple{N3}), make_dense_writable_matrix_from<M33>(2, 3, 4, 6, 7, 8, 10, 11, 12)));

  auto ewd3 = Eigen3::EigenWrapper {Eigen::DiagonalMatrix<double, 3>{make_dense_writable_matrix_from<M31>(1, 2, 3)}};

  EXPECT_TRUE(is_near(get_block(ewd3, std::tuple{N0, N0}, std::tuple{N2, N2}), make_dense_writable_matrix_from<M22>(1, 0, 0, 2)));
  EXPECT_TRUE(is_near(get_block(ewd3, std::tuple{0, 0}, std::tuple{2, 2}), make_dense_writable_matrix_from<M22>(1, 0, 0, 2)));
  EXPECT_TRUE(is_near(get_block<0>(ewd3, std::tuple{N1}, std::tuple{N2}), make_dense_writable_matrix_from<M23>(0, 2, 0, 0, 0, 3)));
  EXPECT_TRUE(is_near(get_block<1>(ewd3, std::tuple{N1}, std::tuple{N2}), make_dense_writable_matrix_from<M32>(0, 0, 2, 0, 0, 3)));

  EXPECT_TRUE(is_near(get_block(Eigen3::EigenWrapper {Eigen::DiagonalMatrix<double, 3>{make_dense_writable_matrix_from<M31>(1, 2, 3)}}, std::tuple{N0, N0}, std::tuple{N2, N2}), make_dense_writable_matrix_from<M22>(1, 0, 0, 2)));
  auto d3a = Eigen::DiagonalMatrix<double, 3>{make_dense_writable_matrix_from<M31>(1, 2, 3)};
  EXPECT_TRUE(is_near(get_block(Eigen3::EigenWrapper {d3a}, std::tuple{N0, N0}, std::tuple{N2, N2}), make_dense_writable_matrix_from<M22>(1, 0, 0, 2)));
  auto ewd3a = Eigen3::EigenWrapper {d3a};
  EXPECT_TRUE(is_near(get_block(ewd3a, std::tuple{N0, N0}, std::tuple{N2, N2}), make_dense_writable_matrix_from<M22>(1, 0, 0, 2)));
}


TEST(eigen3, set_block)
{
  auto N0 = std::integral_constant<std::size_t, 0>{};
  auto N1 = std::integral_constant<std::size_t, 1>{};
  auto N2 = std::integral_constant<std::size_t, 2>{};
  auto N3 = std::integral_constant<std::size_t, 3>{};

  auto m34 = make_dense_writable_matrix_from<M34>(
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12);

  auto ewm34 = Eigen3::EigenWrapper {m34};

  auto m31 = make_dense_writable_matrix_from<M31>(13, 14, 15);

  set_block(m34, m31, N0, N1);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 3, 4,
    5, 14, 7, 8,
    9, 15, 11, 12)));

  set_block(ewm34, m31, N0, 2);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 4,
    5, 14, 14, 8,
    9, 15, 15, 12)));

  auto m14 = make_dense_writable_matrix_from<M14>(16, 17, 18, 19);

  set_block(m34, m14, 1, N0);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 4,
    16, 17, 18, 19,
    9, 15, 15, 12)));

  set_block(m34, m14, 2, 0);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 4,
    16, 17, 18, 19,
    16, 17, 18, 19)));

  auto m21 = make_dense_writable_matrix_from<M21>(20, 21);

  set_block(m34, m21, N1, N1);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 4,
    16, 20, 18, 19,
    16, 21, 18, 19)));

  set_block(m34, m21, 0, N3);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 20,
    16, 20, 18, 21,
    16, 21, 18, 19)));

  auto m13 = make_dense_writable_matrix_from<M13>(22, 23, 24);

  set_block(m34, m13, N2, 1);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 20,
    16, 20, 18, 21,
    16, 22, 23, 24)));

  set_block(m34, m13, 1, 0);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 13, 13, 20,
    22, 23, 24, 21,
    16, 22, 23, 24)));

  const auto m34_copy = m34;
  set_block(m34, get_block(m34, std::tuple{N1, N1}, std::tuple{N1, N2}), N1, N1); EXPECT_TRUE(is_near(m34, m34_copy));
  set_block(m34, get_block(m34, std::tuple{N1, N1}, std::tuple{N1, N2}), 1, 1); EXPECT_TRUE(is_near(m34, m34_copy));
  set_block(m34, get_block(m34, std::tuple{1, 1}, std::tuple{1, 2}), 1, 1); EXPECT_TRUE(is_near(m34, m34_copy));
}


TEST(eigen3, get_chip)
{
  auto m33 = make_dense_writable_matrix_from<M33>(
    1, 7, 4,
    5, 2, 8,
    9, 6, 3);

  std::integral_constant<std::size_t, 0> N0;
  std::integral_constant<std::size_t, 1> N1;
  std::integral_constant<std::size_t, 2> N2;

  auto r0 = make_dense_writable_matrix_from<M13>(1, 7, 4);
  auto r1 = make_dense_writable_matrix_from<M13>(5, 2, 8);
  auto r2 = make_dense_writable_matrix_from<M13>(9, 6, 3);

  auto c0 = make_dense_writable_matrix_from<M31>(1, 5, 9);
  auto c1 = make_dense_writable_matrix_from<M31>(7, 2, 6);
  auto c2 = make_dense_writable_matrix_from<M31>(4, 8, 3);

  EXPECT_TRUE(is_near(get_chip<0>(m33, N1), r1));
  EXPECT_TRUE(is_near(get_chip<1>(m33, 1), c1));
  EXPECT_TRUE(is_near(get_chip<0, 1>(m33, 0, N1), make_dense_writable_matrix_from<M11>(7)));
  EXPECT_TRUE(is_near(get_chip<1, 0>(m33, N2, 1), make_dense_writable_matrix_from<M11>(8)));
  EXPECT_TRUE(is_near(get_chip<0, 1>(m33, N2, N0), make_dense_writable_matrix_from<M11>(9)));

  EXPECT_TRUE(is_near(get_row(m33, 0), r0));
  EXPECT_TRUE(is_near(get_row(M30 {m33}, N1), r1));
  EXPECT_TRUE(is_near(get_row(M03 {m33}, N2), r2));
  EXPECT_TRUE(is_near(get_row(M00 {m33}, 1), r1));

  EXPECT_TRUE(is_near(get_row(m33.array(), 2), r2));
  EXPECT_TRUE(is_near(get_row(M30 {m33}.array(), N1), r1));
  EXPECT_TRUE(is_near(get_row(M03 {m33}.array(), N0), r0));
  EXPECT_TRUE(is_near(get_row(M00 {m33}.array(), 2), r2));

  static_assert(dimension_size_of_index_is<decltype(get_row(M00 {m33}, N0)), 0, 1>);
  static_assert(dimension_size_of_index_is<decltype(get_row(M00 {m33}.array(), N1)), 0, 1>);
  static_assert(dimension_size_of_index_is<decltype(get_row(M00 {m33}, 1)), 0, 1>);
  static_assert(dimension_size_of_index_is<decltype(get_row(M00 {m33}.array(), 2)), 0, 1>);

  EXPECT_TRUE(is_near(get_column(m33, 2), c2));
  EXPECT_TRUE(is_near(get_column(M30 {m33}, N1), c1));
  EXPECT_TRUE(is_near(get_column(M03 {m33}, N0), c0));
  EXPECT_TRUE(is_near(get_column(M00 {m33}, 2), c2));

  EXPECT_TRUE(is_near(get_column(m33.array(), 1), c1));
  EXPECT_TRUE(is_near(get_column(M30 {m33}.array(), N2), c2));
  EXPECT_TRUE(is_near(get_column(M03 {m33}.array(), N1), c1));
  EXPECT_TRUE(is_near(get_column(M00 {m33}.array(), 0), c0));

  static_assert(dimension_size_of_index_is<decltype(get_column(M00 {m33}, N1)), 1, 1>);
  static_assert(dimension_size_of_index_is<decltype(get_column(M00 {m33}.array(), N2)), 1, 1>);
  static_assert(dimension_size_of_index_is<decltype(get_column(M00 {m33}, 2)), 1, 1>);
  static_assert(dimension_size_of_index_is<decltype(get_column(M00 {m33}.array(), 0)), 1, 1>);
}


TEST(eigen3, set_chip)
{
  auto m34 = make_dense_writable_matrix_from<M34>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  std::integral_constant<std::size_t, 0> N0;
  std::integral_constant<std::size_t, 1> N1;
  std::integral_constant<std::size_t, 2> N2;
  std::integral_constant<std::size_t, 3> N3;

  set_chip<0>(m34, make_dense_writable_matrix_from<M14>(1, 2, 3, 4), N0);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 2, 3, 4,
    0, 0, 0, 0,
    0, 0, 0, 0)));
  set_chip<0>(m34, make_dense_writable_matrix_from<M14>(1.5, 2.5, 3.5, 4.5), 1);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 2, 3, 4,
    1.5, 2.5, 3.5, 4.5,
    0, 0, 0, 0)));
  set_chip<0>(m34, make_dense_writable_matrix_from<M14>(5, 6, 7, 8), N2);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    1, 2, 3, 4,
    1.5, 2.5, 3.5, 4.5,
    5, 6, 7, 8)));
  set_chip<1>(m34, make_dense_writable_matrix_from<M31>(9, 10, 11), 0);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    9, 2, 3, 4,
    10, 2.5, 3.5, 4.5,
    11, 6, 7, 8)));
  set_chip<1>(m34, make_dense_writable_matrix_from<M31>(12, 13, 14), N3);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    9, 2, 3, 12,
    10, 2.5, 3.5, 13,
    11, 6, 7, 14)));
  set_row(m34, make_dense_writable_matrix_from<M14>(15, 16, 17, 18), N1);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    9, 2, 3, 12,
    15, 16, 17, 18,
    11, 6, 7, 14)));
  set_column(m34, make_dense_writable_matrix_from<M31>(19, 20, 21), 1);
  EXPECT_TRUE(is_near(m34, make_dense_writable_matrix_from<M34>(
    9, 19, 3, 12,
    15, 20, 17, 18,
    11, 21, 7, 14)));
}


TEST(eigen3, tile)
{
  auto m12a = make_dense_writable_matrix_from<M12>(1, 2);
  auto m12b = make_dense_writable_matrix_from<M12>(3, 4);
  auto m21a = make_dense_writable_matrix_from<M21>(1, 2);
  auto m21b = make_dense_writable_matrix_from<M21>(3, 4);

  EXPECT_TRUE(is_near(tile(std::tuple{Dimensions<2>{}, Dimensions<2>{}}, m12a, m12b), make_dense_writable_matrix_from<M22>(1, 2, 3, 4)));
  EXPECT_TRUE(is_near(tile(std::tuple{2, 2}, m21a, m21b), make_dense_writable_matrix_from<M22>(1, 3, 2, 4)));
  EXPECT_TRUE(is_near(tile(std::tuple{Dimensions<4>{}, Dimensions<2>{}}, m21a, m21b, m12a, m12b), make_dense_writable_matrix_from<M42>(1, 3, 2, 4, 1, 2, 3, 4)));
  EXPECT_TRUE(is_near(tile(std::tuple{Dimensions<4>{}, Dimensions<2>{}}, m12a, m12b, m21a, m21b), make_dense_writable_matrix_from<M42>(1, 2, 3, 4, 1, 3, 2, 4)));
  EXPECT_TRUE(is_near(tile(std::tuple{3, 4}, m12a, m12b, m21a, m21b, m21a, m21b), make_dense_writable_matrix_from<M34>(1, 2, 3, 4, 1, 3, 1, 3, 2, 4, 2, 4)));
}


TEST(eigen3, concatenate_vertical)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m12_56 = make_dense_writable_matrix_from<M12>(5, 6);
  auto m32 = make_dense_writable_matrix_from<M32>(1, 2, 3, 4, 5, 6);

  EXPECT_TRUE(is_near(concatenate<0>(m22, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M20 {m22}, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M20 {m22}, M10 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M20 {m22}, M02 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M20 {m22}, M00 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M02 {m22}, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M02 {m22}, M10 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M02 {m22}, M02 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M02 {m22}, M00 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M00 {m22}, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M00 {m22}, M10 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M00 {m22}, M02 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate<0>(M00 {m22}, M00 {m12_56}), m32));
}


TEST(eigen3, concatenate_horizontal)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 4, 5);
  auto m21_56 = make_dense_writable_matrix_from<M21>(3, 6);
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);

  EXPECT_TRUE(is_near(concatenate<1>(m22, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M20 {m22}, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M20 {m22}, M20 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M20 {m22}, M01 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M20 {m22}, M00 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M02 {m22}, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M02 {m22}, M20 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M02 {m22}, M01 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M02 {m22}, M00 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M00 {m22}, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M00 {m22}, M20 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M00 {m22}, M01 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate<1>(M00 {m22}, M00 {m21_56}), m23));
}


// concatenate_diagonal is in special_matrices tests because it involves constructing zero matrices.


TEST(eigen3, split_vertical)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 0, 0, 2);
  EXPECT_TRUE(is_near(split<0>(m22), std::tuple {}));
  EXPECT_TRUE(is_near(split<0>(M20 {m22}), std::tuple {}));
  EXPECT_TRUE(is_near(split<0>(M02 {m22}), std::tuple {}));
  EXPECT_TRUE(is_near(split<0>(M00 {m22}), std::tuple {}));

  auto x1 = make_eigen_matrix<double, 5, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3,
    4, 0, 0,
    0, 5, 0);

  auto m33 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto tup_m33_m23 = std::tuple {m33, make_eigen_matrix<double, 2, 3>(
    4, 0, 0,
    0, 5, 0)};

  auto tup_m23_m23 = std::tuple {make_eigen_matrix<double, 2, 3>(
    1, 0, 0,
    0, 2, 0), make_eigen_matrix<double, 2, 3>(
    0, 0, 3,
    4, 0, 0)};

  EXPECT_TRUE(is_near(split<0>(x1, std::tuple{Dimensions<3>{}}, std::tuple{Dimensions<2>{}}), tup_m33_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, 5, 3> {x1}, std::tuple{Dimensions<3>{}}, std::tuple{Dimensions<2>{}}), tup_m33_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, 5, dynamic_size> {x1}, Dimensions<3>{}, Dimensions<2>{}), tup_m33_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, dynamic_size, 3> {x1}, Dimensions{3}, Dimensions{2}), tup_m33_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, dynamic_size, dynamic_size> {x1}, 3, 2), tup_m33_m23));

  EXPECT_TRUE(is_near(split<0>(x1, Dimensions<2>{}, Dimensions<2>{}), tup_m23_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, 5, 3> {x1}, Dimensions<2>{}, Dimensions<2>{}), tup_m23_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, 5, dynamic_size> {x1}, Dimensions<2>{}, Dimensions<2>{}), tup_m23_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, dynamic_size, 3> {x1}, std::tuple{Dimensions{2}}, std::tuple{Dimensions{2}}), tup_m23_m23));
  EXPECT_TRUE(is_near(split<0>(eigen_matrix_t<double, dynamic_size, dynamic_size> {x1}, std::tuple{2}, std::tuple{2}), tup_m23_m23));
}


TEST(eigen3, split_horizontal)
{
  auto m22_12 = make_eigen_matrix<double, 2, 2>(
    1, 0,
    0, 2);

  EXPECT_TRUE(is_near(split<1>(m22_12), std::tuple {}));
  EXPECT_TRUE(is_near(split<1>(M22 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split<1>(M20 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split<1>(M02 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split<1>(M00 {m22_12}), std::tuple {}));

  const auto b1 = make_eigen_matrix<double, 3, 5>(
    1, 0, 0, 0, 0,
    0, 2, 0, 4, 0,
    0, 0, 3, 0, 5);

  auto m33 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto tup_m33_m32 = std::tuple {m33, make_eigen_matrix<double, 3, 2>(
    0, 0,
    4, 0,
    0, 5)};

  EXPECT_TRUE(is_near(split<1>(b1, Dimensions<3>{}, Dimensions<2>{}), tup_m33_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, 3, 5> {b1}, Dimensions<3>{}, Dimensions<2>{}), tup_m33_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, 3, dynamic_size> {b1}, Dimensions<3>{}, Dimensions<2>{}), tup_m33_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, dynamic_size, 5> {b1}, std::tuple{Dimensions{3}}, std::tuple{Dimensions{2}}), tup_m33_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, dynamic_size, dynamic_size> {b1}, 3, 2), tup_m33_m32));

  auto tup_m32_m32 = std::tuple {make_eigen_matrix<double, 3, 2>(
    1, 0,
    0, 2,
    0, 0), make_eigen_matrix<double, 3, 2>(
    0, 0,
    0, 4,
    3, 0)};

  EXPECT_TRUE(is_near(split<1>(b1, std::tuple{Dimensions<2>{}}, std::tuple{Dimensions<2>{}}), tup_m32_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, 3, 5>(b1), Dimensions<2>{}, Dimensions<2>{}), tup_m32_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, 3, dynamic_size>(b1), Dimensions<2>{}, Dimensions<2>{}), tup_m32_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, dynamic_size, 5>(b1), Dimensions{2}, Dimensions{2}), tup_m32_m32));
  EXPECT_TRUE(is_near(split<1>(eigen_matrix_t<double, dynamic_size, dynamic_size>(b1), std::tuple{2}, std::tuple{2}), tup_m32_m32));
}


TEST(eigen3, split_diagonal_symmetric)
{
  auto m22_12 = make_eigen_matrix<double, 2, 2>(
    1, 2,
    4, 5);

  EXPECT_TRUE(is_near(split<0, 1>(m22_12), std::tuple {}));
  EXPECT_TRUE(is_near(split<0, 1>(M22 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split<0, 1>(M20 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split<0, 1>(M02 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split<0, 1>(M00 {m22_12}), std::tuple {}));

  auto m55 = make_eigen_matrix<double, 5, 5>(
    1, 2, 3, 0, 0,
    4, 5, 6, 0, 0,
    7, 8, 9, 0, 0,
    0, 0, 0, 10, 11,
    0, 0, 0, 12, 13);

  auto m33 = make_eigen_matrix<double, 3, 3>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9);

  auto tup_m33_m22 = std::tuple {m33, make_eigen_matrix<double, 2, 2>(
    10, 11,
    12, 13)};

  EXPECT_TRUE(is_near(split<0, 1>(m55, Dimensions<3>{}, Dimensions<2>{}), tup_m33_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M55 {m55}, 3, 2), tup_m33_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M50 {m55}, 3, Dimensions<2>{}), tup_m33_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M05 {m55}, Dimensions<3>{}, 2), tup_m33_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M00 {m55}, Dimensions<3>{}, Dimensions<2>{}), tup_m33_m22));

  auto tup_m22_m22 = std::tuple {m22_12, make_eigen_matrix<double, 2, 2>(
    9, 0,
    0, 10)};

  EXPECT_TRUE(is_near(split<0, 1>(m55, Dimensions<2>{}, Dimensions<2>{}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M55 {m55}, Dimensions<2>{}, Dimensions<2>{}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M50 {m55}, Dimensions<2>{}, Dimensions<2>{}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M05 {m55}, Dimensions{2}, Dimensions{2}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M00 {m55}, 2, 2), tup_m22_m22));
}


TEST(eigen3, split_diagonal_asymmetric)
{
  auto m45 = make_eigen_matrix<double, 4, 5>(
    1, 2, 3, 0, 0,
    4, 5, 6, 0, 0,
    0, 0, 0, 7, 8,
    0, 0, 0, 9, 10);

  auto m23 = make_eigen_matrix<double, 2, 3>(
    1, 2, 3,
    4, 5, 6);

  auto tup_m23_m22 = std::tuple {m23, make_eigen_matrix<double, 2, 2>(
    7, 8,
    9, 10)};

  EXPECT_TRUE(is_near(split<0, 1>(m45, std::tuple{Dimensions<2>{}, Dimensions<3>{}}, std::tuple {Dimensions<2>{}, Dimensions<2>{}}), tup_m23_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M45 {m45}, std::tuple{2, 3}, std::tuple {Dimensions<2>{}, 2}), tup_m23_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M40 {m45}, std::tuple{Dimensions<2>{}, 3}, std::tuple {Dimensions<2>{}, Dimensions<2>{}}), tup_m23_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M05 {m45}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}, std::tuple {Dimensions<2>{}, 2}), tup_m23_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M00 {m45}, std::tuple{2, 3}, std::tuple {2, 2}), tup_m23_m22));

  auto m12_12 = make_eigen_matrix<double, 1, 2>(1, 2);

  auto tup_m22_m22 = std::tuple {m12_12, make_eigen_matrix<double, 2, 2>(
    6, 0,
    0, 7)};

  EXPECT_TRUE(is_near(split<0, 1>(m45, std::tuple{1, 2}, std::tuple{2, 2}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M45 {m45}, std::tuple{Dimensions<1>{}, 2}, std::tuple{Dimensions<2>{}, Dimensions{2}}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M40 {m45}, std::tuple{1, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, 2}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M05 {m45}, std::tuple{Dimensions<1>{}, 2}, std::tuple{2, Dimensions{2}}), tup_m22_m22));
  EXPECT_TRUE(is_near(split<0, 1>(M00 {m45}, std::tuple{1, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions{2}}), tup_m22_m22));
}


TEST(eigen3, set_triangle)
{
  const auto a33 = make_dense_writable_matrix_from<M33>(
    1, 2, 3,
    2, 4, 5,
    3, 5, 6);
  const auto b33 = make_dense_writable_matrix_from<M33>(
    1.5, 2.5, 3.5,
    2.5, 4.5, 5.5,
    3.5, 5.5, 6.5);
  const auto l33 = make_dense_writable_matrix_from<M33>(
    1.5, 2, 3,
    2.5, 4.5, 5,
    3.5, 5.5, 6.5);
  const auto u33 = make_dense_writable_matrix_from<M33>(
    1.5, 2.5, 3.5,
    2, 4.5, 5.5,
    3, 5, 6.5);
  const Eigen::TriangularView<const M33, Eigen::Lower> bl33 {b33};
  const Eigen::TriangularView<const M33, Eigen::Upper> bu33 {b33};

  M33 a;

  a = a33; internal::set_triangle<TriangleType::lower>(a, b33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle<TriangleType::upper>(a, b33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle<TriangleType::lower>(a.triangularView<Eigen::Lower>(), b33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle<TriangleType::upper>(a.triangularView<Eigen::Upper>(), b33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle(a.triangularView<Eigen::Lower>(), b33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle(a.triangularView<Eigen::Upper>(), b33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle<TriangleType::lower>(EigenWrapper {a.triangularView<Eigen::Lower>()}, b33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle<TriangleType::upper>(EigenWrapper {a.triangularView<Eigen::Upper>()}, b33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle<TriangleType::lower>(a.selfadjointView<Eigen::Lower>(), b33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle<TriangleType::upper>(a.selfadjointView<Eigen::Upper>(), b33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle<TriangleType::lower>(EigenWrapper {a.selfadjointView<Eigen::Lower>()}, b33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle<TriangleType::upper>(EigenWrapper {a.selfadjointView<Eigen::Upper>()}, b33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle(a, bl33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle(a, bu33);
  EXPECT_TRUE(is_near(a, u33));

  a = a33; internal::set_triangle(a.triangularView<Eigen::Lower>(), bl33);
  EXPECT_TRUE(is_near(a, l33));

  a = a33; internal::set_triangle(a.triangularView<Eigen::Upper>(), bu33);
  EXPECT_TRUE(is_near(a, u33));

  const auto d33 = make_dense_writable_matrix_from<M33>(
    1.5, 2, 3,
    2, 4.5, 5,
    3, 5, 6.5);
  const auto d31 = make_dense_writable_matrix_from<M31>(1, 4, 6);
  const auto e31 = make_dense_writable_matrix_from<M31>(1.5, 4.5, 6.5);
  const auto e33 = e31.asDiagonal();

  a = a33; internal::set_triangle<TriangleType::diagonal>(a, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<TriangleType::diagonal>(a33, b33), d33));

  a = a33; internal::set_triangle(a, e33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle(a33, e33), d33));

  M31 d;

  d = d31; internal::set_triangle(Eigen::DiagonalWrapper<M31>{d}, b33);
  EXPECT_TRUE(is_near(d, e31));
  EXPECT_TRUE(is_near(internal::set_triangle(Eigen::DiagonalMatrix<double, 3>{d31}, b33), e33));

  d = d31; internal::set_triangle(Eigen::DiagonalWrapper<M31>{d}, e33);
  EXPECT_TRUE(is_near(d, e31));
  EXPECT_TRUE(is_near(internal::set_triangle(Eigen::DiagonalMatrix<double, 3>{d31}, e33), e33));
}

