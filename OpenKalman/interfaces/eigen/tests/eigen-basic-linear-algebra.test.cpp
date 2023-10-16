/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, transpose_adjoint_conjugate_matrix)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  auto m32 = make_dense_writable_matrix_from<M32>(1, 4, 2, 5, 3, 6);

  EXPECT_TRUE(is_near(transpose(m23), m32));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(m23)), m32));
  EXPECT_TRUE(is_near(transpose(m23.array()), m32));
  EXPECT_TRUE(is_near(transpose(m2x_3), m32));
  EXPECT_TRUE(is_near(transpose(mx3_2), m32));
  EXPECT_TRUE(is_near(transpose(mxx_23), m32));

  EXPECT_TRUE(is_near(adjoint(m23), m32));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(m23)), m32));
  EXPECT_TRUE(is_near(adjoint(m23.array()), m32));
  EXPECT_TRUE(is_near(adjoint(mx3_2), m32));
  EXPECT_TRUE(is_near(adjoint(m2x_3), m32));
  EXPECT_TRUE(is_near(adjoint(mxx_23), m32));

  EXPECT_TRUE(is_near(conjugate(m23), m23));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(m23)), m23));
  EXPECT_TRUE(is_near(conjugate(m23.array()), m23));
  EXPECT_TRUE(is_near(conjugate(mx3_2), m23));
  EXPECT_TRUE(is_near(conjugate(m2x_3), m23));
  EXPECT_TRUE(is_near(conjugate(mxx_23), m23));

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32 = make_dense_writable_matrix_from<CM32>(cdouble {1,6}, cdouble {4,3}, cdouble {2,5}, cdouble {5,2}, cdouble {3,4}, cdouble {6,1});
  auto cm32conj = make_dense_writable_matrix_from<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(transpose(cm23), cm32));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(cm23)), cm32));
  EXPECT_TRUE(is_near(transpose(cm23.array()), cm32));
  EXPECT_TRUE(is_near(adjoint(cm23), cm32conj));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(cm23)), cm32conj));
  EXPECT_TRUE(is_near(adjoint(cm23.array()), cm32conj));
  EXPECT_TRUE(is_near(conjugate(cm32), cm32conj));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm32)), cm32conj));
  EXPECT_TRUE(is_near(conjugate(cm32.array()), cm32conj));
}


TEST(eigen3, transpose_adjoint_conjugate_self_adjoint)
{
  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M2x {m22_93310};
  auto m02_93310 = Mx2 {m22_93310};
  auto m00_93310 = Mxx {m22_93310};

  EXPECT_TRUE(is_near(transpose(m22_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Lower>())), m22_93310));
  EXPECT_TRUE(is_near(transpose(m20_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m02_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m00_93310.template selfadjointView<Eigen::Upper>()), m22_93310));

  EXPECT_TRUE(is_near(adjoint(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Upper>())), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  EXPECT_TRUE(is_near(conjugate(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Upper>())), m22_93310));
  EXPECT_TRUE(is_near(conjugate(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(conjugate(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(conjugate(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  auto cm22_93310 = make_dense_writable_matrix_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);
  auto cm22_93310_trans = make_dense_writable_matrix_from<CM22>(9, cdouble(3,-1), cdouble(3,1), 10);

  EXPECT_TRUE(is_near(transpose(cm22_93310.template selfadjointView<Eigen::Upper>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(transpose(std::as_const(cm22_93310).template selfadjointView<Eigen::Upper>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(cm22_93310.template selfadjointView<Eigen::Upper>())), cm22_93310_trans));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(std::as_const(cm22_93310).template selfadjointView<Eigen::Upper>())), cm22_93310_trans));
  EXPECT_TRUE(is_near(adjoint(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(cm22_93310.template selfadjointView<Eigen::Lower>())), cm22_93310));
  EXPECT_TRUE(is_near(conjugate(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm22_93310.template selfadjointView<Eigen::Lower>())), cm22_93310_trans));
}



TEST(eigen3, transpose_adjoint_conjugate_triangular)
{
  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M2x {m22_3103};
  auto m02_3103 = Mx2 {m22_3103};
  auto m00_3103 = Mxx {m22_3103};

  auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M2x {m22_3013};
  auto m02_3013 = Mx2 {m22_3013};
  auto m00_3013 = Mxx {m22_3013};

  EXPECT_TRUE(is_near(transpose(m22_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(m22_3013.template triangularView<Eigen::Lower>())), m22_3103));
  EXPECT_TRUE(is_near(transpose(m20_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(transpose(m02_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(transpose(m00_3103.template triangularView<Eigen::Upper>()), m22_3013));

  EXPECT_TRUE(is_near(adjoint(m22_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(m22_3103.template triangularView<Eigen::Upper>())), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m20_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(adjoint(m02_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m00_3013.template triangularView<Eigen::Lower>()), m22_3103));

  EXPECT_TRUE(is_near(conjugate(m22_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(m22_3103.template triangularView<Eigen::Upper>())), m22_3103));
  EXPECT_TRUE(is_near(conjugate(m20_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(conjugate(m02_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(conjugate(m00_3013.template triangularView<Eigen::Lower>()), m22_3013));

  auto cm22_3013 = make_dense_writable_matrix_from<CM22>(cdouble(3,0.3), 0, cdouble(1,0.1), 3);
  auto cm22_3103 = make_dense_writable_matrix_from<CM22>(cdouble(3,0.3), cdouble(1,0.1), 0, 3);
  auto cm22_3013_conj = make_dense_writable_matrix_from<CM22>(cdouble(3,-0.3), 0, cdouble(1,-0.1), 3);
  auto cm22_3103_conj = make_dense_writable_matrix_from<CM22>(cdouble(3,-0.3), cdouble(1,-0.1), 0, 3);

  EXPECT_TRUE(is_near(transpose(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013));
  EXPECT_TRUE(is_near(transpose(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3103));
  EXPECT_TRUE(is_near(transpose(std::as_const(cm22_3013).template triangularView<Eigen::Lower>()), cm22_3103));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(cm22_3103.template triangularView<Eigen::Upper>())), cm22_3013));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(std::as_const(cm22_3103).template triangularView<Eigen::Upper>())), cm22_3013));

  EXPECT_TRUE(is_near(adjoint(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013_conj));
  EXPECT_TRUE(is_near(adjoint(std::as_const(cm22_3103).template triangularView<Eigen::Upper>()), cm22_3013_conj));
  EXPECT_TRUE(is_near(adjoint(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3103_conj));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(cm22_3013.template triangularView<Eigen::Lower>())), cm22_3103_conj));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(std::as_const(cm22_3013).template triangularView<Eigen::Lower>())), cm22_3103_conj));

  EXPECT_TRUE(is_near(conjugate(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3013_conj));
  EXPECT_TRUE(is_near(conjugate(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3103_conj));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm22_3013.template triangularView<Eigen::Lower>())), cm22_3013_conj));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm22_3103.template triangularView<Eigen::Upper>())), cm22_3103_conj));
}


TEST(eigen3, transpose_adjoint_conjugate_diagonal)
{
  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mx1_2 = Mx1 {m21};
  auto mxx_21 = Mxx {m21};

  auto m22_1004 = make_dense_writable_matrix_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21})), m22_1004));

  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));

  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));
}


TEST(eigen3, determinant_trace)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);

  EXPECT_NEAR(determinant(m22), -2, 1e-6);
  EXPECT_NEAR(determinant(m22.array()), -2, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(m22)), -2, 1e-6);
  EXPECT_NEAR(determinant(M2x {m22}), -2, 1e-6);
  EXPECT_NEAR(determinant(Mx2 {m22}), -2, 1e-6);
  EXPECT_NEAR(determinant(Mxx {m22}), -2, 1e-6);

  const M00 m00 {};
  EXPECT_NEAR(determinant(m00), 1, 1e-6);
  EXPECT_NEAR(determinant(M0x {m00}), 1, 1e-6);
  EXPECT_NEAR(determinant(Mx0 {m00}), 1, 1e-6);
  EXPECT_NEAR(determinant(Mxx {m00}), 1, 1e-6);

  EXPECT_NEAR(determinant(M11{2}), 2, 1e-6);
  EXPECT_NEAR(determinant((M1x(1,1) << 2).finished()), 2, 1e-6);
  EXPECT_NEAR(determinant((Mx1(1,1) << 2).finished()), 2, 1e-6);
  EXPECT_NEAR(determinant((Mxx(1,1) << 2).finished()), 2, 1e-6);

  EXPECT_NEAR(trace(m22), 5, 1e-6);
  EXPECT_NEAR(trace(m22.array()), 5, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(m22)), 5, 1e-6);
  EXPECT_NEAR(trace(M2x {m22}), 5, 1e-6);
  EXPECT_NEAR(trace(Mx2 {m22}), 5, 1e-6);
  EXPECT_NEAR(trace(Mxx {m22}), 5, 1e-6);

  EXPECT_NEAR(trace(m00), 0, 1e-6);
  //EXPECT_NEAR(trace(M0x{m00}), 0, 1e-6); // Tested with special matrices
  //EXPECT_NEAR(trace(Mx0{m00}), 0, 1e-6); // Tested with special matrices
  EXPECT_NEAR(trace(Mxx{m00}), 0, 1e-6);

  EXPECT_NEAR(trace(M11{3}), 3, 1e-6);
  EXPECT_NEAR(trace((M1x(1,1) << 3).finished()), 3, 1e-6);
  EXPECT_NEAR(trace((Mx1(1,1) << 3).finished()), 3, 1e-6);
  EXPECT_NEAR(trace((Mxx(1,1) << 3).finished()), 3, 1e-6);

  auto cm22 = make_dense_writable_matrix_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});
  auto cm22_3103 = make_dense_writable_matrix_from<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);

  EXPECT_NEAR(std::real(determinant(cm22)), 0, 1e-6);
  EXPECT_NEAR(std::real(determinant(cm22.array())), 0, 1e-6);
  EXPECT_NEAR(std::real(determinant(Eigen3::make_eigen_wrapper(cm22))), 0, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22)), 5, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22.array())), 5, 1e-6);
  EXPECT_NEAR(std::real(trace(Eigen3::make_eigen_wrapper(cm22))), 5, 1e-6);

  EXPECT_NEAR(std::imag(determinant(cm22)), 4, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22.array())), 4, 1e-6);
  EXPECT_NEAR(std::imag(determinant(Eigen3::make_eigen_wrapper(cm22))), 4, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22)), 5, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22.array())), 5, 1e-6);
  EXPECT_NEAR(std::imag(trace(Eigen3::make_eigen_wrapper(cm22))), 5, 1e-6);

  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);

  EXPECT_NEAR(determinant(m22_93310.template selfadjointView<Eigen::Lower>()), 81, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Lower>())), 81, 1e-6);
  EXPECT_NEAR(trace(m22_93310.template selfadjointView<Eigen::Lower>()), 19, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);

  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);

  EXPECT_NEAR(determinant(m22_3103.template triangularView<Eigen::Upper>()), 9, 1e-6);
  EXPECT_NEAR(trace(m22_3103.template triangularView<Eigen::Upper>()), 6, 1e-6);

  auto m21 = M21 {1, 4};

  EXPECT_NEAR(determinant(Eigen::DiagonalMatrix<double, 2> {m21}), 4, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), 4, 1e-6);
  EXPECT_NEAR(trace(Eigen::DiagonalMatrix<double, 2> {m21}), 5, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), 5, 1e-6);

  EXPECT_NEAR(determinant(Eigen::DiagonalWrapper<M21> {m21}), 4, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), 4, 1e-6);
  EXPECT_NEAR(trace(Eigen::DiagonalWrapper<M21> {m21}), 5, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), 5, 1e-6);

  auto cm22_93310 = make_dense_writable_matrix_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);

  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Lower>())), 80, 1e-6);
  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Upper>())), 80, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Upper>())), 19, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22_3103.template triangularView<Eigen::Upper>())), 9, 1e-6);
  EXPECT_NEAR(std::real(determinant(cm22_3103.template triangularView<Eigen::Lower>())), 9, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3103.template triangularView<Eigen::Upper>())), -3, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3103.template triangularView<Eigen::Lower>())), -3, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22_3103.template triangularView<Eigen::Lower>())), 6, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_3103.template triangularView<Eigen::Upper>())), 6, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3103.template triangularView<Eigen::Lower>())), -1, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3103.template triangularView<Eigen::Upper>())), -1, 1e-6);
}


TEST(eigen3, sum)
{
  auto m23a = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m23b = make_dense_writable_matrix_from<M23>(7, 8, 9, 10, 11, 12);
  auto m23c = make_dense_writable_matrix_from<M23>(8, 10, 12, 14, 16, 18);

  M2x m2x_3a {m23a};
  M2x m2x_3b {m23b};
  Mx3 mx3_2a {m23a};
  Mx3 mx3_2b {m23b};
  Mxx mxx_23a {m23a};
  Mxx mxx_23b {m23b};

  EXPECT_TRUE(is_near(sum(m23a, m23b), m23c));
  EXPECT_TRUE(is_near(sum(m2x_3a, mx3_2b), m23c));
  EXPECT_TRUE(is_near(sum(mx3_2a, m2x_3b), m23c));
  EXPECT_TRUE(is_near(sum(mxx_23a, mxx_23b), m23c));
  static_assert(not dynamic_dimension<decltype(sum(m2x_3a, mx3_2b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(m2x_3a, mx3_2b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(m2x_3a, m2x_3b)), 0>);
  static_assert(dynamic_dimension<decltype(sum(m2x_3a, m2x_3b)), 1>);
  static_assert(dynamic_dimension<decltype(sum(mx3_2a, mx3_2b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(mx3_2a, mx3_2b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(mx3_2a, m2x_3b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(mx3_2a, m2x_3b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(mxx_23a, m23b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(mxx_23a, m23b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(m23a, mxx_23b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(m23a, mxx_23b)), 1>);
}


TEST(eigen3, contract)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};

  auto m34 = make_dense_writable_matrix_from<M34>(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
  auto m3x_4 {m34};
  auto mx4_3 {m34};
  auto mxx_34 {m34};

  auto m24 = make_dense_writable_matrix_from<M24>(74, 80, 86, 92, 173, 188, 203, 218);

  EXPECT_TRUE(is_near(contract(m23, m34), m24));
  EXPECT_TRUE(is_near(contract(m23.array(), m34.array()), m24));
  EXPECT_TRUE(is_near(contract(Eigen3::make_eigen_wrapper(m23), Eigen3::make_eigen_wrapper(m34)), m24));
  EXPECT_TRUE(is_near(contract(M23{m23}, M34{m34}), m24));
  EXPECT_TRUE(is_near(contract(m23, M34{m34}), m24));
  EXPECT_TRUE(is_near(contract(M23{m23}, m34), m24));
  EXPECT_TRUE(is_near(contract(m2x_3, m34), m24));
  EXPECT_TRUE(is_near(contract(m2x_3, mx4_3), m24));
  EXPECT_TRUE(is_near(contract(m23, mx4_3), m24));
  EXPECT_TRUE(is_near(contract(mx3_2, m3x_4), m24));

  EXPECT_TRUE(is_near(contract(m23, M33::Identity()), m23));
  EXPECT_TRUE(is_near(contract(M22::Identity(), m23), m23));

  auto m11_2 = make_dense_writable_matrix_from<M11>(2);
  M1x m10_1_2(1,1); m10_1_2 << 2;
  Mx1 m01_1_2(1,1); m01_1_2 << 2;
  Mxx m00_11_2(1,1); m00_11_2 << 2;

  auto m11_4 = make_dense_writable_matrix_from<M11>(4);

  auto m11_5 = make_dense_writable_matrix_from<M11>(5);
  M1x m10_1_5(1,1); m10_1_5 << 5;
  Mx1 m01_1_5(1,1); m01_1_5 << 5;
  Mxx m00_11_5(1,1); m00_11_5 << 5;

  auto m11_10 = make_dense_writable_matrix_from<M11>(10);

  EXPECT_TRUE(is_near(contract(m11_2, m11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_2, m11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_2, m11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_2, m11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m11_2, m10_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_2, m10_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_2, m10_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_2, m10_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m11_2, m01_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_2, m01_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_2, m01_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_2, m01_1_5), m11_10));
  EXPECT_TRUE(is_near(contract(m11_2, m00_11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_2, m00_11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_2, m00_11_5), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_2, m00_11_5), m11_10));

  auto c11_2 {M11::Identity() + M11::Identity()};
  auto c10_1_2 = Eigen::Replicate<decltype(c11_2), 1, Eigen::Dynamic>(c11_2, 1, 1);
  auto c01_1_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1>(c11_2, 1, 1);
  auto c00_11_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 1, 1);

  EXPECT_TRUE(is_near(contract(c11_2, m11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c10_1_2, m11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c01_1_2, m11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c00_11_2, m11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c11_2, m10_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c10_1_2, m10_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c01_1_2, m10_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c00_11_2, m10_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c11_2, m01_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c10_1_2, m01_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c01_1_2, m01_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c00_11_2, m01_1_2), m11_4));
  EXPECT_TRUE(is_near(contract(c11_2, m00_11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c10_1_2, m00_11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c01_1_2, m00_11_2), m11_4));
  EXPECT_TRUE(is_near(contract(c00_11_2, m00_11_2), m11_4));

  EXPECT_TRUE(is_near(contract(m11_5, c11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m11_5, c10_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m11_5, c01_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m11_5, c00_11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_5, c11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_5, c10_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_5, c01_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m10_1_5, c00_11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_5, c11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_5, c10_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_5, c01_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m01_1_5, c00_11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_5, c11_2), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_5, c10_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_5, c01_1_2), m11_10));
  EXPECT_TRUE(is_near(contract(m00_11_5, c00_11_2), m11_10));

  auto m31a = make_dense_writable_matrix_from<M31>(2, 3, 4);
  auto m01_3a = Mx1{m31a};
  auto m31b = make_dense_writable_matrix_from<M31>(5, 6, 7);
  auto m01_3b = Mx1{m31b};

  auto dm3a = Eigen::DiagonalMatrix<double, 3>{m31a};
  auto dm0_3a = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31a};
  auto dw3a = Eigen::DiagonalWrapper{m31a};

  auto dm3b = Eigen::DiagonalMatrix<double, 3>{m31b};
  auto dm0_3b = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31b};
  auto dw3b = Eigen::DiagonalWrapper{m31b};

  M33 d3c {make_dense_writable_matrix_from<M31>(10, 18, 28).asDiagonal()};

  auto c23_2 = Eigen::Replicate<decltype(c11_2), 2, 3>(c11_2);
  auto c20_3_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 3);
  auto c03_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 3>(c11_2, 2, 3);
  auto c00_23_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 3);

  auto m23_468 = make_dense_writable_matrix_from<M23>(4, 6, 8, 4, 6, 8);

  EXPECT_TRUE(is_near(contract(c23_2, dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(c20_3_2, dw3a), m23_468));
  EXPECT_TRUE(is_near(contract(c03_2_2, dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(c00_23_2, dw3a), m23_468));

  auto c11_3 {M11::Identity() + M11::Identity() + M11::Identity()};
  auto c33_3 = Eigen::Replicate<decltype(c11_3), 3, 3>(c11_3);
  auto c30_3_3 = Eigen::Replicate<decltype(c11_3), 3, Eigen::Dynamic>(c11_3, 3, 3);
  auto c03_3_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, 3>(c11_3, 3, 3);
  auto c00_33_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, Eigen::Dynamic>(c11_3, 3, 3);

  auto m23_151821 = make_dense_writable_matrix_from<M33>(15, 15, 15, 18, 18, 18, 21, 21, 21);

  EXPECT_TRUE(is_near(contract(dw3b, c33_3), m23_151821));
  EXPECT_TRUE(is_near(contract(dm3b, c30_3_3), m23_151821));
  EXPECT_TRUE(is_near(contract(dw3b, c03_3_3), m23_151821));
  EXPECT_TRUE(is_near(contract(dm3b, c00_33_3), m23_151821));
}
