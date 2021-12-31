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
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_extent, dynamic_extent>;
  using M10 = eigen_matrix_t<double, 1, dynamic_extent>;
  using M01 = eigen_matrix_t<double, dynamic_extent, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_extent>;
  using M02 = eigen_matrix_t<double, dynamic_extent, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_extent>;
  using M03 = eigen_matrix_t<double, dynamic_extent, 3>;
  using M04 = eigen_matrix_t<double, dynamic_extent, 4>;
  using M50 = eigen_matrix_t<double, 5, dynamic_extent>;
  using M05 = eigen_matrix_t<double, dynamic_extent, 5>;

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
  auto m23 = make_native_matrix<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  auto m32 = make_native_matrix<M32>(1, 4, 2, 5, 3, 6);

  EXPECT_TRUE(is_near(transpose(m23), m32));
  EXPECT_TRUE(is_near(transpose(m20_3), m32));
  EXPECT_TRUE(is_near(transpose(m03_2), m32));
  EXPECT_TRUE(is_near(transpose(m00_23), m32));

  EXPECT_TRUE(is_near(adjoint(m23), m32));
  EXPECT_TRUE(is_near(adjoint(m03_2), m32));
  EXPECT_TRUE(is_near(adjoint(m20_3), m32));
  EXPECT_TRUE(is_near(adjoint(m00_23), m32));

  auto cm23 = make_native_matrix<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32 = make_native_matrix<CM32>(cdouble {1,6}, cdouble {4,3}, cdouble {2,5}, cdouble {5,2}, cdouble {3,4}, cdouble {6,1});
  auto cm32conj = make_native_matrix<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(transpose(cm23), cm32));
  EXPECT_TRUE(is_near(adjoint(cm23), cm32conj));
}


TEST(eigen3, transpose_adjoint_self_adjoint)
{
  auto m22_93310 = make_native_matrix<M22>(9, 3, 3, 10);
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

  auto cm22_93310 = make_native_matrix<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);
  auto cm22_93310_trans = make_native_matrix<CM22>(9, cdouble(3,-1), cdouble(3,1), 10);

  EXPECT_TRUE(is_near(transpose(cm22_93310.template selfadjointView<Eigen::Upper>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(adjoint(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310));
}



TEST(eigen3, transpose_adjoint_triangular)
{
  auto m22_3103 = make_native_matrix<M22>(3, 1, 0, 3);
  auto m20_3103 = M20 {m22_3103};
  auto m02_3103 = M02 {m22_3103};
  auto m00_3103 = M00 {m22_3103};

  auto m22_3013 = make_native_matrix<M22>(3, 0, 1, 3);
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

  auto cm22_3013 = make_native_matrix<CM22>(cdouble(3,1), 0, cdouble(1,1), 3);
  auto cm22_3103 = make_native_matrix<CM22>(cdouble(3,1), cdouble(1,1), 0, 3);
  auto cm22_3103_conj = make_native_matrix<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);

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

  auto m22_1004 = make_native_matrix<M22>(1, 0, 0, 4);

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


TEST(eigen3, transpose_adjoint_zero)
{
  auto z11 {M11::Identity() - M11::Identity()};

  auto z21 {(M22::Identity() - M22::Identity()).diagonal()};
  auto z01_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 1> {z11, 2, 1};
  auto z20_1 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 1};
  auto z00_21 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 1};

  auto z12 = Eigen::Replicate<decltype(z11), 1, 2> {z11, 1, 2};

  EXPECT_TRUE(is_near(transpose(z21), z12)); static_assert(zero_matrix<decltype(transpose(z21))>);
  EXPECT_TRUE(is_near(transpose(z20_1), z12)); static_assert(zero_matrix<decltype(transpose(z20_1))>);
  EXPECT_TRUE(is_near(transpose(z01_2), z12)); static_assert(zero_matrix<decltype(transpose(z01_2))>);
  EXPECT_TRUE(is_near(transpose(z00_21), z12)); static_assert(zero_matrix<decltype(transpose(z00_21))>);

  EXPECT_TRUE(is_near(adjoint(z21), z12)); static_assert(zero_matrix<decltype(adjoint(z21))>);
  EXPECT_TRUE(is_near(adjoint(z20_1), z12)); static_assert(zero_matrix<decltype(adjoint(z20_1))>);
  EXPECT_TRUE(is_near(adjoint(z01_2), z12)); static_assert(zero_matrix<decltype(adjoint(z01_2))>);
  EXPECT_TRUE(is_near(adjoint(z00_21), z12)); static_assert(zero_matrix<decltype(adjoint(z00_21))>);
}


TEST(eigen3, transpose_adjoint_constant)
{
  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c21_2 = c11_2.replicate<2, 1>();
  auto c20_1_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 1);
  auto c01_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1>(c11_2, 2, 1);
  auto c00_21_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 1);

  auto c12_2 = Eigen::Replicate<decltype(c11_2), 1, 2> {c11_2, 1, 2};

  EXPECT_TRUE(is_near(transpose(c21_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c21_2))> == 2);
  EXPECT_TRUE(is_near(transpose(c20_1_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c20_1_2))> == 2);
  EXPECT_TRUE(is_near(transpose(c01_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c01_2_2))> == 2);
  EXPECT_TRUE(is_near(transpose(c00_21_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c00_21_2))> == 2);

  EXPECT_TRUE(is_near(adjoint(c21_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c21_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(c20_1_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c20_1_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(c01_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c01_2_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(c00_21_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c00_21_2))> == 2);
}


TEST(eigen3, determinant_trace)
{
  auto m22 = make_native_matrix<M22>(1, 2, 3, 4);

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

  auto cm22 = make_native_matrix<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});
  auto cm22_3103 = make_native_matrix<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);

  EXPECT_NEAR(std::real(determinant(cm22)), 0, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22)), 5, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22)), 4, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22)), 5, 1e-6);

  auto m22_93310 = make_native_matrix<M22>(9, 3, 3, 10);

  EXPECT_NEAR(determinant(m22_93310.template selfadjointView<Eigen::Lower>()), 81, 1e-6);
  EXPECT_NEAR(trace(m22_93310.template selfadjointView<Eigen::Lower>()), 19, 1e-6);

  auto m22_3103 = make_native_matrix<M22>(3, 1, 0, 3);

  EXPECT_NEAR(determinant(m22_3103.template triangularView<Eigen::Upper>()), 9, 1e-6);
  EXPECT_NEAR(trace(m22_3103.template triangularView<Eigen::Upper>()), 6, 1e-6);

  auto m21 = M21 {1, 4};

  EXPECT_NEAR(determinant(Eigen::DiagonalMatrix<double, 2> {m21}), 4, 1e-6);
  EXPECT_NEAR(trace(Eigen::DiagonalMatrix<double, 2> {m21}), 5, 1e-6);
  EXPECT_NEAR(determinant(Eigen::DiagonalWrapper<M21> {m21}), 4, 1e-6);
  EXPECT_NEAR(trace(Eigen::DiagonalWrapper<M21> {m21}), 5, 1e-6);

  auto cm22_93310 = make_native_matrix<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);

  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Lower>())), 80, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22_3103.template triangularView<Eigen::Upper>())), 9, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3103.template triangularView<Eigen::Upper>())), -3, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22_3103.template triangularView<Eigen::Upper>())), 6, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3103.template triangularView<Eigen::Upper>())), -1, 1e-6);

  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_NEAR(determinant(z22), 0, 1e-6);
  EXPECT_NEAR(determinant(z20_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z02_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z00_22), 0, 1e-6);

  EXPECT_NEAR(trace(z22), 0, 1e-6);
  EXPECT_NEAR(trace(z20_2), 0, 1e-6);
  EXPECT_NEAR(trace(z02_2), 0, 1e-6);
  EXPECT_NEAR(trace(z00_22), 0, 1e-6);

  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c22_2 = c11_2.replicate<2, 2>();
  auto c20_2_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 2);
  auto c02_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2>(c11_2, 2, 2);
  auto c00_22_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 2);

  EXPECT_NEAR(determinant(c22_2), 0, 1e-6);
  EXPECT_NEAR(determinant(c20_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(c02_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(c00_22_2), 0, 1e-6);

  EXPECT_NEAR(trace(c22_2), 4, 1e-6);
  EXPECT_NEAR(trace(c20_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(c02_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(c00_22_2), 4, 1e-6);

  EXPECT_NEAR(determinant(M22::Identity()), 1, 1e-6);
  EXPECT_NEAR(trace(M22::Identity()), 2, 1e-6);
}
