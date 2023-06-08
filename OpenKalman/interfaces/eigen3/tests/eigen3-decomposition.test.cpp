/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2022 Christopher Lee Ogden <ogden@gatech.edu>
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
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M99 = eigen_matrix_t<double, 9, 9>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M40 = eigen_matrix_t<double, 4, dynamic_size>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
  using M90 = eigen_matrix_t<double, 9, dynamic_size>;
  using M09 = eigen_matrix_t<double, dynamic_size, 9>;

  using cdouble = std::complex<double>;

  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
}


TEST(eigen3, LQ_and_QR_decomp_constant)
{
  auto z11 = M11::Identity() - M11::Identity();
  auto z22 = M22::Identity() - M22::Identity();
  auto z33 = M33::Identity() - M33::Identity();
  auto z23 = z11.replicate<2, 3>();
  auto z32 = z11.replicate<3, 2>();
  EXPECT_TRUE(is_near(LQ_decomposition(z23), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z32), z33));
  EXPECT_TRUE(is_near(QR_decomposition(z23), z33));
  EXPECT_TRUE(is_near(QR_decomposition(z32), z22));

  auto c11 = M11::Identity() + M11::Identity();
  auto l44 = make_dense_writable_matrix_from<M44>(
    6, 0, 0, 0,
    6, 0, 0, 0,
    6, 0, 0, 0,
    6, 0, 0, 0);
  auto u44 = make_dense_writable_matrix_from<M44>(
    6, 6, 6, 6,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0);
  auto l99 = make_dense_writable_matrix_from<M99>(
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0);
  auto u99 = make_dense_writable_matrix_from<M99>(
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0);

  auto c49 = c11.replicate<4, 9>();
  auto c40_9 = Eigen::Replicate<decltype(c11), 4, Eigen::Dynamic> {c11, 4, 9};
  auto c09_4 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, 9> {c11, 4, 9};
  auto c00_49 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 4, 9};

  auto c94 = c11.replicate<9, 4>();
  auto c90_4 = Eigen::Replicate<decltype(c11), 9, Eigen::Dynamic> {c11, 9, 4};
  auto c04_9 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, 4> {c11, 9, 4};
  auto c00_94 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 9, 4};

  EXPECT_TRUE(is_near(LQ_decomposition(c49), l44));
  EXPECT_TRUE(is_near(LQ_decomposition(c40_9), l44));
  EXPECT_TRUE(is_near(LQ_decomposition(c09_4), l44));
  EXPECT_TRUE(is_near(LQ_decomposition(c00_49), l44));

  EXPECT_TRUE(is_near(LQ_decomposition(c94), l99));
  EXPECT_TRUE(is_near(LQ_decomposition(c90_4), l99));
  EXPECT_TRUE(is_near(LQ_decomposition(c04_9), l99));
  EXPECT_TRUE(is_near(LQ_decomposition(c00_94), l99));

  EXPECT_TRUE(is_near(QR_decomposition(c49), u99));
  EXPECT_TRUE(is_near(QR_decomposition(c40_9), u99));
  EXPECT_TRUE(is_near(QR_decomposition(c09_4), u99));
  EXPECT_TRUE(is_near(QR_decomposition(c00_49), u99));

  EXPECT_TRUE(is_near(QR_decomposition(c94), u44));
  EXPECT_TRUE(is_near(QR_decomposition(c90_4), u44));
  EXPECT_TRUE(is_near(QR_decomposition(c04_9), u44));
  EXPECT_TRUE(is_near(QR_decomposition(c00_94), u44));
}


TEST(eigen3, LQ_and_QR_decomp_general)
{
  auto m22_lq = make_dense_writable_matrix_from<M22>(-0.1, 0, 1.096, -1.272);
  EXPECT_TRUE(is_near(LQ_decomposition(m22_lq.triangularView<Eigen::Lower>()), m22_lq));
  EXPECT_TRUE(is_near(QR_decomposition(adjoint(m22_lq).triangularView<Eigen::Upper>()), adjoint(m22_lq)));

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

  EXPECT_TRUE(is_near(LQ_decomposition(m23), adjoint(make_dense_writable_matrix_from(QR_decomposition(m32)))));
  EXPECT_TRUE(is_near(LQ_decomposition(m23), adjoint(QR_decomposition(m32))));
  EXPECT_TRUE(is_near(adjoint(LQ_decomposition(m23)), QR_decomposition(m32)));
  EXPECT_TRUE(is_near(LQ_decomposition(m32), adjoint(QR_decomposition(m23))));
  EXPECT_TRUE(is_near(adjoint(LQ_decomposition(m32)), QR_decomposition(m23)));
}


TEST(eigen3, LQ_and_QR_decomp_complex)
{
  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32conj = make_dense_writable_matrix_from<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(LQ_decomposition(cm23), adjoint(QR_decomposition(cm32conj))));
  EXPECT_TRUE(is_near(adjoint(LQ_decomposition(cm23)), QR_decomposition(cm32conj)));
  EXPECT_TRUE(is_near(LQ_decomposition(cm32conj), adjoint(QR_decomposition(cm23))));
  EXPECT_TRUE(is_near(adjoint(LQ_decomposition(cm32conj)), QR_decomposition(cm23)));
}

