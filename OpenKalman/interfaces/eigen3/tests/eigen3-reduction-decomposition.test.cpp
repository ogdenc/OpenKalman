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


TEST(eigen3, reduce_columns_reduce_rows_matrix)
{
  EXPECT_TRUE(is_near(reduce_columns(M21 {1, 4}), M21 {1, 4}));
  EXPECT_TRUE(is_near(reduce_rows(make_eigen_matrix<double, 1, 3>(1, 2, 3)), make_eigen_matrix<double, 1, 3>(1, 2, 3)));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  auto m21_25 = make_eigen_matrix<double, 2, 1>(2, 5);

  EXPECT_TRUE(is_near(reduce_columns(m23), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(m03_2), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(m20_3), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(m00_23), m21_25));

  auto i20_2 = M20::Identity(2, 2);
  auto i02_2 = M02::Identity(2, 2);
  auto i00_22 = M00::Identity(2, 2);

  EXPECT_TRUE(is_near(reduce_columns(i20_2), M21::Constant(0.5)));
  auto rci02_2 = reduce_columns(i02_2);
  EXPECT_TRUE(is_near(rci02_2, M21::Constant(0.5)));
  EXPECT_TRUE(is_near(reduce_columns(i02_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(reduce_columns(i00_22), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(reduce_columns(M20::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(reduce_columns(M02::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(reduce_columns(M00::Identity(2, 2)), M21::Constant(0.5)));

  auto m13_234 = make_eigen_matrix<double, 1, 3>(2.5, 3.5, 4.5);

  EXPECT_TRUE(is_near(reduce_rows(m23), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(m03_2), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(m20_3), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(m00_23), m13_234));

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});

  EXPECT_TRUE(is_near(reduce_columns(cm23), make_eigen_matrix<cdouble, 2, 1>(cdouble {2,5}, cdouble {5,2})));

  EXPECT_TRUE(is_near(reduce_rows(cm23), make_eigen_matrix<cdouble, 1, 3>(cdouble {2.5,4.5}, cdouble{3.5,3.5}, cdouble {4.5,2.5})));
}


TEST(eigen3, reduce_columns_reduce_rows_zero)
{
  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  auto z21 = (M22::Identity() - M22::Identity()).diagonal();

  EXPECT_TRUE(is_near(reduce_columns(z22), z21)); static_assert(zero_matrix<decltype(reduce_columns(z22))>);
  EXPECT_TRUE(is_near(reduce_columns(z20_2), z21)); static_assert(zero_matrix<decltype(reduce_columns(z20_2))>);
  EXPECT_TRUE(is_near(reduce_columns(z02_2), z21)); static_assert(zero_matrix<decltype(reduce_columns(z02_2))>);
  EXPECT_TRUE(is_near(reduce_columns(z00_22), z21)); static_assert(zero_matrix<decltype(reduce_columns(z00_22))>);

  auto z12 = Eigen::Replicate<decltype(z11), 1, 2> {z11, 1, 2};

  EXPECT_TRUE(is_near(reduce_rows(z22), z12)); static_assert(zero_matrix<decltype(reduce_rows(z22))>);
  EXPECT_TRUE(is_near(reduce_rows(z20_2), z12)); static_assert(zero_matrix<decltype(reduce_rows(z20_2))>);
  EXPECT_TRUE(is_near(reduce_rows(z02_2), z12)); static_assert(zero_matrix<decltype(reduce_rows(z02_2))>);
  EXPECT_TRUE(is_near(reduce_rows(z00_22), z12)); static_assert(zero_matrix<decltype(reduce_rows(z00_22))>);
}


TEST(eigen3, reduce_columns_reduce_rows_constant)
{
  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c22_2 = c11_2.replicate<2, 2>();
  auto c20_2_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 2);
  auto c02_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2>(c11_2, 2, 2);
  auto c00_22_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 2);

  EXPECT_TRUE(is_near(reduce_columns(c22_2), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_columns(c22_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(c20_2_2), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_columns(c20_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(c02_2_2), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_columns(c02_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(c00_22_2), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_columns(c00_22_2))> == 2);

  EXPECT_TRUE(is_near(reduce_rows(c22_2), M12::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_rows(c22_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(c20_2_2), M12::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_rows(c20_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(c02_2_2), M12::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_rows(c02_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(c00_22_2), M12::Constant(2))); static_assert(constant_coefficient_v<decltype(reduce_rows(c00_22_2))> == 2);
}


TEST(eigen3, reduce_columns_reduce_rows_diagonal)
{
  auto i22 = M22::Identity();
  auto i20_2 = Eigen::Replicate<typename M11::IdentityReturnType, 2, Eigen::Dynamic> {M11::Identity(), 2, 1}.asDiagonal();
  auto i02_2 = Eigen::Replicate<typename M11::IdentityReturnType, Eigen::Dynamic, 1> {M11::Identity(), 2, 1}.asDiagonal();
  auto i00_22 = Eigen::Replicate<typename M11::IdentityReturnType, Eigen::Dynamic, Eigen::Dynamic> {M11::Identity(), 2, 1}.asDiagonal();

  EXPECT_TRUE(is_near(reduce_columns(i22 + i22), M21::Constant(1))); static_assert(are_within_tolerance(constant_coefficient_v<decltype(reduce_columns(i22 + i22))>, 1));
  EXPECT_TRUE(is_near(reduce_columns(i20_2), M21::Constant(0.5))); static_assert(not constant_matrix<decltype(reduce_columns(i20_2))>);
  EXPECT_TRUE(is_near(reduce_columns(i02_2), M21::Constant(0.5))); static_assert(not constant_matrix<decltype(reduce_columns(i02_2))>);
  EXPECT_TRUE(is_near(reduce_columns(i00_22), M21::Constant(0.5))); static_assert(not constant_matrix<decltype(reduce_columns(i00_22))>);

  EXPECT_TRUE(is_near(reduce_rows(i22 + i22), M12::Constant(1))); static_assert(are_within_tolerance(constant_coefficient_v<decltype(reduce_rows(i22 + i22))>, 1));
  EXPECT_TRUE(is_near(reduce_rows(i20_2), M12::Constant(0.5))); static_assert(not constant_matrix<decltype(reduce_rows(i20_2))>);
  EXPECT_TRUE(is_near(reduce_rows(i02_2), M12::Constant(0.5))); static_assert(not constant_matrix<decltype(reduce_rows(i02_2))>);
  EXPECT_TRUE(is_near(reduce_rows(i00_22), M12::Constant(0.5))); static_assert(not constant_matrix<decltype(reduce_rows(i00_22))>);

  auto c11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = c11_2.replicate<2, 1>().asDiagonal();
  auto d20_1_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();
  auto d01_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1> {c11_2, 2, 1}.asDiagonal();
  auto d00_21_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();

  EXPECT_TRUE(is_near(reduce_columns(d21_2), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(reduce_columns(d21_2))> == 1);
  EXPECT_TRUE(is_near(reduce_columns(d20_1_2), M21::Constant(1))); static_assert(not constant_matrix<decltype(reduce_columns(d20_1_2))>);
  EXPECT_TRUE(is_near(reduce_columns(d01_2_2), M21::Constant(1))); static_assert(not constant_matrix<decltype(reduce_columns(d01_2_2))>);
  EXPECT_TRUE(is_near(reduce_columns(d00_21_2), M21::Constant(1))); static_assert(not constant_matrix<decltype(reduce_columns(d00_21_2))>);

  EXPECT_TRUE(is_near(reduce_rows(d21_2), M12::Constant(1))); static_assert(constant_coefficient_v<decltype(reduce_rows(d21_2))> == 1);
  EXPECT_TRUE(is_near(reduce_rows(d20_1_2), M12::Constant(1))); static_assert(not constant_matrix<decltype(reduce_rows(d20_1_2))>);
  EXPECT_TRUE(is_near(reduce_rows(d01_2_2), M12::Constant(1))); static_assert(not constant_matrix<decltype(reduce_rows(d01_2_2))>);
  EXPECT_TRUE(is_near(reduce_rows(d00_21_2), M12::Constant(1))); static_assert(not constant_matrix<decltype(reduce_rows(d00_21_2))>);
}


TEST(eigen3, LQ_and_QR_decompositions)
{
  auto m22_lq = make_dense_writable_matrix_from<M22>(-0.1, 0, 1.096, -1.272);

  auto m22_lq_decomp = make_dense_writable_matrix_from<M22>(0.06, 0.08, 0.36, -1.640);
  auto m20_2_lq_decomp = M20 {m22_lq_decomp};
  auto m02_2_lq_decomp = M02 {m22_lq_decomp};
  auto m00_22_lq_decomp = M00 {m22_lq_decomp};

  EXPECT_TRUE(is_near(LQ_decomposition(m22_lq_decomp), m22_lq));
  EXPECT_TRUE(is_near(LQ_decomposition(m20_2_lq_decomp), m22_lq));
  EXPECT_TRUE(is_near(LQ_decomposition(m02_2_lq_decomp), m22_lq));
  EXPECT_TRUE(is_near(LQ_decomposition(m00_22_lq_decomp), m22_lq));

  auto m22_qr = make_dense_writable_matrix_from<M22>(-0.1, 1.096, 0, -1.272);

  auto m22_qr_decomp = make_dense_writable_matrix_from<M22>(0.06, 0.36, 0.08, -1.640);
  auto m20_2_qr_decomp = M20 {m22_qr_decomp};
  auto m02_2_qr_decomp = M02 {m22_qr_decomp};
  auto m00_22_qr_decomp = M00 {m22_qr_decomp};

  EXPECT_TRUE(is_near(QR_decomposition(m22_qr_decomp), m22_qr));
  EXPECT_TRUE(is_near(QR_decomposition(m02_2_qr_decomp), m22_qr));
  EXPECT_TRUE(is_near(QR_decomposition(m20_2_qr_decomp), m22_qr));
  EXPECT_TRUE(is_near(QR_decomposition(m00_22_qr_decomp), m22_qr));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m32 = make_dense_writable_matrix_from<M32>(1, 4, 2, 5, 3, 6);

  EXPECT_TRUE(is_near(LQ_decomposition(m23), adjoint(QR_decomposition(m32))));
  EXPECT_TRUE(is_near(LQ_decomposition(m32), adjoint(QR_decomposition(m23))));

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32conj = make_dense_writable_matrix_from<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(LQ_decomposition(cm23), adjoint(QR_decomposition(cm32conj))));
  EXPECT_TRUE(is_near(LQ_decomposition(cm32conj), adjoint(QR_decomposition(cm23))));
}

