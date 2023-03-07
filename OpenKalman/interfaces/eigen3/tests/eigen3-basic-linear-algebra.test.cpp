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
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M24 = eigen_matrix_t<double, 2, 4>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
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


TEST(eigen3, transpose_adjoint_matrix)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  auto m32 = make_dense_writable_matrix_from<M32>(1, 4, 2, 5, 3, 6);

  EXPECT_TRUE(is_near(transpose(m23), m32));
  EXPECT_TRUE(is_near(transpose(m20_3), m32));
  EXPECT_TRUE(is_near(transpose(m03_2), m32));
  EXPECT_TRUE(is_near(transpose(m00_23), m32));

  EXPECT_TRUE(is_near(adjoint(m23), m32));
  EXPECT_TRUE(is_near(adjoint(m03_2), m32));
  EXPECT_TRUE(is_near(adjoint(m20_3), m32));
  EXPECT_TRUE(is_near(adjoint(m00_23), m32));

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32 = make_dense_writable_matrix_from<CM32>(cdouble {1,6}, cdouble {4,3}, cdouble {2,5}, cdouble {5,2}, cdouble {3,4}, cdouble {6,1});
  auto cm32conj = make_dense_writable_matrix_from<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(transpose(cm23), cm32));
  EXPECT_TRUE(is_near(adjoint(cm23), cm32conj));
}


TEST(eigen3, transpose_adjoint_self_adjoint)
{
  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M20 {m22_93310};
  auto m02_93310 = M02 {m22_93310};
  auto m00_93310 = M00 {m22_93310};

  EXPECT_TRUE(is_near(transpose(m22_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m20_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m02_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m00_93310.template selfadjointView<Eigen::Upper>()), m22_93310));

  EXPECT_TRUE(is_near(adjoint(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  auto cm22_93310 = make_dense_writable_matrix_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);
  auto cm22_93310_trans = make_dense_writable_matrix_from<CM22>(9, cdouble(3,-1), cdouble(3,1), 10);

  EXPECT_TRUE(is_near(transpose(cm22_93310.template selfadjointView<Eigen::Upper>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(adjoint(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310));
}



TEST(eigen3, transpose_adjoint_triangular)
{
  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M20 {m22_3103};
  auto m02_3103 = M02 {m22_3103};
  auto m00_3103 = M00 {m22_3103};

  auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M20 {m22_3013};
  auto m02_3013 = M02 {m22_3013};
  auto m00_3013 = M00 {m22_3013};

  EXPECT_TRUE(is_near(transpose(m22_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(transpose(m20_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(transpose(m02_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(transpose(m00_3103.template triangularView<Eigen::Upper>()), m22_3013));

  EXPECT_TRUE(is_near(adjoint(m22_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m20_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(adjoint(m02_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m00_3013.template triangularView<Eigen::Lower>()), m22_3103));

  auto cm22_3013 = make_dense_writable_matrix_from<CM22>(cdouble(3,1), 0, cdouble(1,1), 3);
  auto cm22_3103 = make_dense_writable_matrix_from<CM22>(cdouble(3,1), cdouble(1,1), 0, 3);
  auto cm22_3103_conj = make_dense_writable_matrix_from<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);

  EXPECT_TRUE(is_near(transpose(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013));
  EXPECT_TRUE(is_near(transpose(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3103));
  EXPECT_TRUE(is_near(adjoint(cm22_3103_conj.template triangularView<Eigen::Upper>()), cm22_3013));
  EXPECT_TRUE(is_near(adjoint(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3103_conj));
}


TEST(eigen3, transpose_adjoint_diagonal)
{
  auto m21 = M21 {1, 4};
  auto m20_1 = M20 {m21};
  auto m01_2 = M01 {m21};
  auto m00_21 = M00 {m21};

  auto m22_1004 = make_dense_writable_matrix_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M20> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M01> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M00> {m00_21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M20> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M01> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M00> {m00_21}), m22_1004));
}


TEST(eigen3, determinant_trace)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);

  EXPECT_NEAR(determinant(M11 {2}), 2, 1e-6);
  EXPECT_NEAR(determinant(m22), -2, 1e-6);
  EXPECT_NEAR(determinant(M20 {m22}), -2, 1e-6);
  EXPECT_NEAR(determinant(M02 {m22}), -2, 1e-6);
  EXPECT_NEAR(determinant(M00 {m22}), -2, 1e-6);

  EXPECT_NEAR(trace(M11 {3}), 3, 1e-6);
  EXPECT_NEAR(trace(m22), 5, 1e-6);
  EXPECT_NEAR(trace(M20 {m22}), 5, 1e-6);
  EXPECT_NEAR(trace(M02 {m22}), 5, 1e-6);
  EXPECT_NEAR(trace(M00 {m22}), 5, 1e-6);

  auto cm22 = make_dense_writable_matrix_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});
  auto cm22_3103 = make_dense_writable_matrix_from<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);

  EXPECT_NEAR(std::real(determinant(cm22)), 0, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22)), 5, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22)), 4, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22)), 5, 1e-6);

  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);

  EXPECT_NEAR(determinant(m22_93310.template selfadjointView<Eigen::Lower>()), 81, 1e-6);
  EXPECT_NEAR(trace(m22_93310.template selfadjointView<Eigen::Lower>()), 19, 1e-6);

  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);

  EXPECT_NEAR(determinant(m22_3103.template triangularView<Eigen::Upper>()), 9, 1e-6);
  EXPECT_NEAR(trace(m22_3103.template triangularView<Eigen::Upper>()), 6, 1e-6);

  auto m21 = M21 {1, 4};

  EXPECT_NEAR(determinant(Eigen::DiagonalMatrix<double, 2> {m21}), 4, 1e-6);
  EXPECT_NEAR(trace(Eigen::DiagonalMatrix<double, 2> {m21}), 5, 1e-6);
  EXPECT_NEAR(determinant(Eigen::DiagonalWrapper<M21> {m21}), 4, 1e-6);
  EXPECT_NEAR(trace(Eigen::DiagonalWrapper<M21> {m21}), 5, 1e-6);

  auto cm22_93310 = make_dense_writable_matrix_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);

  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Lower>())), 80, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22_3103.template triangularView<Eigen::Upper>())), 9, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3103.template triangularView<Eigen::Upper>())), -3, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22_3103.template triangularView<Eigen::Upper>())), 6, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3103.template triangularView<Eigen::Upper>())), -1, 1e-6);
}


TEST(eigen3, sum)
{
  auto m23a = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m23b = make_dense_writable_matrix_from<M23>(7, 8, 9, 10, 11, 12);
  auto m23c = make_dense_writable_matrix_from<M23>(8, 10, 12, 14, 16, 18);

  EXPECT_TRUE(is_near(sum(m23a, m23b), m23c));
}


TEST(eigen3, contract)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = make_dense_writable_matrix_from<M20>(m23);
  auto m03_2 = make_dense_writable_matrix_from<M03>(m23);
  auto m00_23 = make_dense_writable_matrix_from<M00>(m23);

  auto m34 = make_dense_writable_matrix_from<M34>(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
  auto m30_4 = make_dense_writable_matrix_from<M30>(m34);
  auto m04_3 = make_dense_writable_matrix_from<M04>(m34);
  auto m00_34 = make_dense_writable_matrix_from<M00>(m34);

  auto m24 = make_dense_writable_matrix_from<M24>(74, 80, 86, 92, 173, 188, 203, 218);

  EXPECT_TRUE(is_near(contract(m23, m34), m24));
  EXPECT_TRUE(is_near(contract(M23{m23}, M34{m34}), m24));
  EXPECT_TRUE(is_near(contract(m23, M34{m34}), m24));
  EXPECT_TRUE(is_near(contract(M23{m23}, m34), m24));
  EXPECT_TRUE(is_near(contract(m20_3, m34), m24));
  EXPECT_TRUE(is_near(contract(m20_3, m04_3), m24));
  EXPECT_TRUE(is_near(contract(m23, m04_3), m24));
  EXPECT_TRUE(is_near(contract(m03_2, m30_4), m24));

  EXPECT_TRUE(is_near(contract(m23, M33::Identity()), m23));
  EXPECT_TRUE(is_near(contract(M22::Identity(), m23), m23));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  auto m11_2 = make_dense_writable_matrix_from<M11>(2); auto m10_1_2 = M10{m11_2}; auto m01_1_2 = M01{m11_2}; auto m00_11_2 = M00{m11_2};
  auto m11_4 = make_dense_writable_matrix_from<M11>(4);
  auto m11_5 = make_dense_writable_matrix_from<M11>(5); auto m10_1_5 = M10{m11_5}; auto m01_1_5 = M01{m11_5}; auto m00_11_5 = M00{m11_5};
  auto m11_10 = make_dense_writable_matrix_from<M11>(10);
#pragma GCC diagnostic pop

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
  auto m01_3a = M01{m31a};
  auto m31b = make_dense_writable_matrix_from<M31>(5, 6, 7);
  auto m01_3b = M01{m31b};

  auto dm3a = Eigen::DiagonalMatrix<double, 3>{m31a};
  auto dm0_3a = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31a};
  auto dw3a = Eigen::DiagonalWrapper{m31a};
  auto dw0_3a = Eigen::DiagonalWrapper{m01_3a};

  auto dm3b = Eigen::DiagonalMatrix<double, 3>{m31b};
  auto dm0_3b = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31b};
  auto dw3b = Eigen::DiagonalWrapper{m31b};
  auto dw0_3b = Eigen::DiagonalWrapper{m01_3b};

  auto d3c = DiagonalMatrix<M31> {10, 18, 28};

  EXPECT_TRUE(is_near(contract(dm3a, dm3b), d3c));
  EXPECT_TRUE(is_near(contract(dm0_3a, dm3b), d3c));
  EXPECT_TRUE(is_near(contract(dm3a, dm0_3b), d3c));
  EXPECT_TRUE(is_near(contract(dm0_3a, dm0_3b), d3c));
  EXPECT_TRUE(is_near(contract(dw3a, dw3b), d3c));
  EXPECT_TRUE(is_near(contract(dw0_3a, dw3b), d3c));
  EXPECT_TRUE(is_near(contract(dw3a, dw0_3b), d3c));
  EXPECT_TRUE(is_near(contract(dw0_3a, dw0_3b), d3c));

  auto c23_2 = Eigen::Replicate<decltype(c11_2), 2, 3>(c11_2);
  auto c20_3_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 3);
  auto c03_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 3>(c11_2, 2, 3);
  auto c00_23_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 3);

  auto m23_468 = make_dense_writable_matrix_from<M23>(4, 6, 8, 4, 6, 8);

  EXPECT_TRUE(is_near(contract(c23_2, dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(c20_3_2, dw3a), m23_468));
  EXPECT_TRUE(is_near(contract(c03_2_2, dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(c00_23_2, dw3a), m23_468));
  EXPECT_TRUE(is_near(contract(make_constant_matrix_like<M23, 2>(), dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(make_constant_matrix_like<M23, 2>(), dw3a), m23_468));

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
  EXPECT_TRUE(is_near(contract(dm3b, make_constant_matrix_like<M33, 3>()), m23_151821));
  EXPECT_TRUE(is_near(contract(dw3b, make_constant_matrix_like<M33, 3>()), m23_151821));
}
