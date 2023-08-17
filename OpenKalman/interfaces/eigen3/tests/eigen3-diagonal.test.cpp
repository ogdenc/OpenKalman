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

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, to_diagonal)
{
  auto m11 = M11 {3};
  auto m1x_1 = M1x {m11};
  auto mx1_1 = Mx1 {m11};
  Mxx mxx_11(1,1); mxx_11 << 3;

  // diagonal_matrix input:

  EXPECT_TRUE(is_near(to_diagonal(Eigen::DiagonalWrapper {m11}), m11));
  EXPECT_TRUE(is_near(to_diagonal(Eigen::DiagonalMatrix<double, 1> {m11}), m11));
  EXPECT_TRUE(is_near(to_diagonal(m11.selfadjointView<Eigen::Lower>()), m11));
  EXPECT_TRUE(is_near(to_diagonal(m11.triangularView<Eigen::Upper>()), m11));
  EXPECT_TRUE(is_near(to_diagonal(EigenWrapper {m11.triangularView<Eigen::Upper>()}), m11));

  // constant one-by-one input:

  static_assert(constant_coefficient_v<decltype(to_diagonal(M11::Identity() + M11::Identity()))> == 2);

  // Handled by dense Eigen interface:

  EXPECT_TRUE(is_near(to_diagonal(m11), m11)); static_assert(std::is_same_v<decltype(to_diagonal(m11)), M11&>);
  EXPECT_TRUE(is_near(to_diagonal(m1x_1), m11)); static_assert(not has_dynamic_dimensions<decltype(to_diagonal(m1x_1))>);
  EXPECT_TRUE(is_near(to_diagonal(mx1_1), m11)); static_assert(has_dynamic_dimensions<decltype(to_diagonal(mx1_1))>); static_assert(eigen_DiagonalWrapper<decltype(to_diagonal(mx1_1))>);
  EXPECT_TRUE(is_near(to_diagonal(mxx_11), m11)); static_assert(has_dynamic_dimensions<decltype(to_diagonal(mxx_11))>); static_assert(eigen_DiagonalWrapper<decltype(to_diagonal(mxx_11))>);

  EXPECT_TRUE(is_near(to_diagonal(M11 {m11}), m11)); static_assert(std::is_same_v<decltype(to_diagonal(M11 {m11})), M11&&>);
  EXPECT_TRUE(is_near(to_diagonal(M1x {m11}), m11)); static_assert(not has_dynamic_dimensions<decltype(to_diagonal(M1x {m11}))>);
  // to_diagonal(M1x {m11}) must create OpenKalman::DiagonalMatrix. See special-matrices/tests/DiagonalMatrix.test.cpp
  // to_diagonal(Mxx {m11}) must create OpenKalman::DiagonalMatrix. See special-matrices/tests/DiagonalMatrix.test.cpp
}


TEST(eigen3, diagonal_of_diagonal_adapter)
{
  auto m21 = M21 {1, 4};
  auto mx1_2 = Mx1 {m21};

  auto dm2 = Eigen::DiagonalMatrix<double, 2> {m21}; static_assert(diagonal_adapter<decltype(dm2)>);
  auto dm0_2 = Eigen::DiagonalMatrix<double, Eigen::Dynamic> {mx1_2}; static_assert(diagonal_adapter<decltype(dm0_2)>);

  EXPECT_TRUE(is_near(diagonal_of(dm2), m21));
  EXPECT_TRUE(is_near(diagonal_of(dm0_2), mx1_2));

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalMatrix<double, 2> {dm2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {dm0_2}), m21));

  static_assert(diagonal_adapter<decltype(Eigen::DiagonalWrapper {m21})>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m21}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {mx1_2}), m21));

  static_assert(std::is_assignable_v<decltype(nested_matrix(Eigen::DiagonalWrapper {m21})), M21>);
  static_assert(std::is_assignable_v<decltype(diagonal_of(Eigen::DiagonalWrapper {m21})), M21>);
}


TEST(eigen3, diagonal_of_diagonal_wrapper) // For Eigen::DiagonalWrapper cases not handled by general diagonal_of function.
{
  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mxx_21 = Mxx {m21};

  static_assert(not diagonal_adapter<decltype(Eigen::DiagonalWrapper {m2x_1})>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m2x_1}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {mxx_21}), m21));

  auto m12 = M12 {1, 4};
  auto m1x_2 = M1x {m12};
  auto mx2_1 = Mx2 {m12};
  auto mxx_12 = Mxx {m12};

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m12}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m1x_2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {mx2_1}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {mxx_12}), m21));

  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  const auto m41 = make_dense_writable_matrix_from<M41>(1, 3, 2, 4);

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m22}), m41));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m2x_2}), m41));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {mx2_2}), m41));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {mxx_22}), m41));
}


TEST(eigen3, diagonal_of_1x1)
{
  // fixed one-by-one:

  auto m11 = M11 {3};
  EXPECT_TRUE(is_near(diagonal_of(m11), m11));
  EXPECT_EQ(&diagonal_of(m11), &m11);
  EXPECT_TRUE(is_near(diagonal_of(M11{m11}), m11));
  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(M11::Identity()))>);
  static_assert(constant_coefficient_v<decltype(diagonal_of(M11::Identity()))> == 1);

  // Note: dynamic one-by-one, known at compile time requires creation of ConstantAdapter and is tested elsewhere.

  // dynamic one-by-one, unknown at compile time:

  auto m1x_1 = M1x {m11};
  auto mx1_1 = Mx1 {m11};
  Mxx mxx_11(1,1); mxx_11 = m11;

  EXPECT_TRUE(is_near(diagonal_of(m1x_1), m11)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(m1x_1))>);
  EXPECT_TRUE(is_near(diagonal_of(mx1_1), m11)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(mx1_1))>);
  EXPECT_TRUE(is_near(diagonal_of(mxx_11), m11)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(mxx_11))>);
  static_assert(constant_matrix<decltype(diagonal_of(m11)), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(diagonal_of(m1x_1)), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(diagonal_of(mx1_1)), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(diagonal_of(mxx_11)), CompileTimeStatus::any>);
  static_assert(constant_matrix<decltype(diagonal_of(mxx_11)), CompileTimeStatus::any, Likelihood::maybe>);
}


TEST(eigen3, diagonal_of_constant)
{
  auto c11_3 = M11::Constant(3);
  auto c22_3 = M22::Constant(3);
  auto c21_3 = M21::Constant(3);

  static_assert(constant_matrix<decltype(diagonal_of(c11_3)), CompileTimeStatus::unknown>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(c11_3)), 0, 1>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(c11_3)), 1, 1>);
  EXPECT_TRUE(is_near(diagonal_of(c11_3), c11_3));
  EXPECT_EQ(get_element(diagonal_of(c11_3)), 3);

  static_assert(constant_matrix<decltype(diagonal_of(c22_3)), CompileTimeStatus::unknown>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(c22_3)), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(c22_3)), 1, 1>);
  EXPECT_TRUE(is_near(diagonal_of(c22_3), c21_3));
  EXPECT_EQ(get_element(diagonal_of(c22_3)), 3);

  auto cd22_3 = c21_3.asDiagonal();

  static_assert(constant_matrix<decltype(diagonal_of(cd22_3)), CompileTimeStatus::unknown>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(cd22_3)), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(cd22_3)), 1, 1>);
  EXPECT_TRUE(is_near(diagonal_of(cd22_3), c21_3));
  EXPECT_EQ(get_element(diagonal_of(cd22_3)), 3);
}


TEST(eigen3, diagonal_of_dense)
{
  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mx1_2 = Mx1 {m21};
  auto mxx_21 = Mxx {m21};

  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  auto dm2 = Eigen::DiagonalMatrix<double, 2> {m21};
  auto dm0_2 = Eigen::DiagonalMatrix<double, Eigen::Dynamic> {mx1_2};

  const auto m41 = make_dense_writable_matrix_from<M41>(1, 3, 2, 4);

  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalMatrix<double, 2> {dm2}}), m21));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalMatrix<double, Eigen::Dynamic> {dm0_2}}), m21));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalWrapper {m21}}), m21));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalWrapper {m22}}), m41));

  EXPECT_TRUE(is_near(diagonal_of(m22), m21));
  EXPECT_TRUE(is_near(diagonal_of(m2x_2), m21)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(m2x_2))>);
  EXPECT_TRUE(is_near(diagonal_of(mx2_2), m21)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(mx2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(mxx_22), m21)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(mxx_22))>);
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {m22}), m21));

  EXPECT_TRUE(is_near(diagonal_of(M22 {m22}), m21));
  EXPECT_TRUE(is_near(diagonal_of(M2x {m2x_2}), m21)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(M2x {m2x_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(Mx2 {mx2_2}), m21)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(Mx2 {mx2_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(Mxx {mxx_22}), m21)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(Mxx {mxx_22}))>);
}


TEST(eigen3, diagonal_of_self_adjoint)
{
  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M2x {m22_93310};
  auto m02_93310 = Mx2 {m22_93310};
  auto m00_93310 = Mxx {m22_93310};

  auto m21_910 = make_dense_writable_matrix_from<M21>(9, 10);

  EXPECT_TRUE(is_near(diagonal_of(m22_93310.template selfadjointView<Eigen::Upper>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m20_93310.template selfadjointView<Eigen::Lower>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m02_93310.template selfadjointView<Eigen::Upper>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m00_93310.template selfadjointView<Eigen::Lower>()), m21_910));
  EXPECT_TRUE(is_near(EigenWrapper {diagonal_of(m22_93310.template selfadjointView<Eigen::Upper>())}, m21_910));
}


TEST(eigen3, diagonal_of_triangular)
{
  auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M2x {m22_3013};
  auto m02_3013 = Mx2 {m22_3013};
  auto m00_3013 = Mxx {m22_3013};

  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M2x {m22_3103};
  auto m02_3103 = Mx2 {m22_3103};
  auto m00_3103 = Mxx {m22_3103};

  auto m21_33 = make_dense_writable_matrix_from<M21>(3, 3);

  EXPECT_TRUE(is_near(diagonal_of(m22_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m20_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m02_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m00_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {m22_3013.template triangularView<Eigen::Lower>()}), m21_33));

  EXPECT_TRUE(is_near(diagonal_of(m22_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m20_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m02_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m00_3103.template triangularView<Eigen::Upper>()), m21_33));
}

