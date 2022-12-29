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

