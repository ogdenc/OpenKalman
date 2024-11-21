/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, contract)
{
  auto m23 = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};

  // Identity

  EXPECT_TRUE(is_near(contract(m23, M33::Identity()), m23));
  EXPECT_TRUE(is_near(contract(m2x_3, M33::Identity()), m23)); static_assert(dimension_size_of_index_is<decltype(contract(m2x_3, M33::Identity())), 1, 3>);
  EXPECT_TRUE(is_near(contract(mxx_23, M33::Identity()), m23)); static_assert(dimension_size_of_index_is<decltype(contract(mxx_23, M33::Identity())), 1, 3>);
  EXPECT_TRUE(is_near(contract(M22::Identity(), m23), m23));
  EXPECT_TRUE(is_near(contract(M22::Identity(), mx3_2), m23)); static_assert(dimension_size_of_index_is<decltype(contract(M22::Identity(), mx3_2)), 0, 2>);
  EXPECT_TRUE(is_near(contract(M22::Identity(), mxx_23), m23)); static_assert(dimension_size_of_index_is<decltype(contract(M22::Identity(), mxx_23)), 0, 2>);

  // multiplication by zero matrix is tested elsewhere in the context of ConstantAdapter

  // one-dimensional

  auto m11_2 = make_dense_object_from<M11>(2);
  M1x m10_1_2(1,1); m10_1_2 << 2;
  Mx1 m01_1_2(1,1); m01_1_2 << 2;
  Mxx m00_11_2(1,1); m00_11_2 << 2;

  auto m11_4 = make_dense_object_from<M11>(4);

  auto m11_5 = make_dense_object_from<M11>(5);
  M1x m10_1_5(1,1); m10_1_5 << 5;
  Mx1 m01_1_5(1,1); m01_1_5 << 5;
  Mxx m00_11_5(1,1); m00_11_5 << 5;

  auto m11_10 = make_dense_object_from<M11>(10);

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

  // diagonal * constant

  auto m31b = make_dense_object_from<M31>(5, 6, 7);
  auto dm3b = Eigen::DiagonalMatrix<double, 3>{m31b};
  auto dm0_3b = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31b};
  auto dw3b = Eigen::DiagonalWrapper{m31b};

  auto c11_3 {M11::Identity() + M11::Identity() + M11::Identity()};
  auto c33_3 = Eigen::Replicate<decltype(c11_3), 3, 3>(c11_3);
  auto c30_3_3 = Eigen::Replicate<decltype(c11_3), 3, Eigen::Dynamic>(c11_3, 3, 3);
  auto c03_3_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, 3>(c11_3, 3, 3);
  auto c00_33_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, Eigen::Dynamic>(c11_3, 3, 3);

  auto m23_151821 = make_dense_object_from<M33>(15, 15, 15, 18, 18, 18, 21, 21, 21);

  EXPECT_TRUE(is_near(contract(dw3b, c33_3), m23_151821));
  EXPECT_TRUE(is_near(contract(dm3b, c30_3_3), m23_151821));
  EXPECT_TRUE(is_near(contract(dw3b, c03_3_3), m23_151821));
  EXPECT_TRUE(is_near(contract(dm3b, c00_33_3), m23_151821));

  // constant * diagonal

  auto m31a = make_dense_object_from<M31>(2, 3, 4);
  auto dm3a = Eigen::DiagonalMatrix<double, 3>{m31a};
  auto dm0_3a = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31a};
  auto dw3a = Eigen::DiagonalWrapper{m31a};

  auto c23_2 = Eigen::Replicate<decltype(c11_2), 2, 3>(c11_2);
  auto c20_3_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 3);
  auto c03_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 3>(c11_2, 2, 3);
  auto c00_23_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 3);

  auto m23_468 = make_dense_object_from<M23>(4, 6, 8, 4, 6, 8);

  EXPECT_TRUE(is_near(contract(c23_2, dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(c20_3_2, dw3a), m23_468));
  EXPECT_TRUE(is_near(contract(c03_2_2, dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(c00_23_2, dw3a), m23_468));

  // diagonal * diagonal -- Must create OpenKalman::DiagonalAdapter, and is tested in that context.

  // regular matrices

  auto m34 = make_dense_object_from<M34>(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
  auto m3x_4 {m34};
  auto mx4_3 {m34};
  auto mxx_34 {m34};

  auto m24 = make_dense_object_from<M24>(74, 80, 86, 92, 173, 188, 203, 218);

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

  // diagonal * matrix

  auto m34_14 = make_dense_object_from<M34>(14, 16, 18, 20, 33, 36, 39, 42, 60, 64, 68, 72);

  EXPECT_TRUE(is_near(contract(dm3a, m34), m34_14));
  EXPECT_TRUE(is_near(contract(dw3a, m3x_4), m34_14));
  EXPECT_TRUE(is_near(contract(dw3a, mx4_3), m34_14));
  EXPECT_TRUE(is_near(contract(dm0_3a, mxx_34), m34_14));

  // matrix * diagonal

  auto m23_5 = make_dense_object_from<M23>(5, 12, 21, 20, 30, 42);

  EXPECT_TRUE(is_near(contract(m23, dm3b), m23_5));
  EXPECT_TRUE(is_near(contract(m2x_3, dw3b), m23_5));
  EXPECT_TRUE(is_near(contract(mx3_2, dw3b), m23_5));
  EXPECT_TRUE(is_near(contract(mxx_23, dm0_3b), m23_5));

  // triangular * matrix

  auto m33 = make_dense_object_from<M33>(1, 2, 3, 4, 5, 6, 7, 8, 9);
  auto m3x_3 {m33};
  auto mx3_3 {m33};
  auto mxx_33 {m33};

  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::Upper>(), m34), make_dense_object_from<M34>(74, 80, 86, 92, 145, 156, 167, 178, 135, 144, 153, 162)));
  EXPECT_TRUE(is_near(contract(mxx_33.template triangularView<Eigen::Upper>(), mxx_34), make_dense_object_from<M34>(74, 80, 86, 92, 145, 156, 167, 178, 135, 144, 153, 162)));
  static_assert(dimension_size_of_index_is<decltype(contract(mx3_3.template triangularView<Eigen::Upper>(), m34)), 0, 3>);

  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::Lower>(), m34), make_dense_object_from<M34>(7, 8, 9, 10, 83, 92, 101, 110, 272, 296, 320, 344)));
  EXPECT_TRUE(is_near(contract(mxx_33.template triangularView<Eigen::Lower>(), mxx_34), make_dense_object_from<M34>(7, 8, 9, 10, 83, 92, 101, 110, 272, 296, 320, 344)));
  static_assert(dimension_size_of_index_is<decltype(contract(m33.template triangularView<Eigen::Lower>(), m3x_4)), 1, 4>);

  // matrix * triangular

  EXPECT_TRUE(is_near(contract(m23, m33.template triangularView<Eigen::Upper>()), make_dense_object_from<M23>(1, 12, 42, 4, 33, 96)));
  EXPECT_TRUE(is_near(contract(m23, m33.template triangularView<Eigen::Lower>()), make_dense_object_from<M23>(30, 34, 27, 66, 73, 54)));

  // triangular * diagonal

  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::Upper>(), dm3b), make_dense_object_from<M33>(5, 12, 21, 0, 30, 42, 0, 0, 63)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::Upper>(), dm3b)), TriangleType::upper>);

  // triangular * constant diagonal

  auto c31_2 = Eigen::Replicate<decltype(c11_2), 3, 1>(c11_2);
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::Upper>(), c31_2.asDiagonal()), make_dense_object_from<M33>(2, 4, 6, 0, 10, 12, 0, 0, 18)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::Upper>(), c31_2.asDiagonal())), TriangleType::upper>);

  // diagonal * triangular

  EXPECT_TRUE(is_near(contract(dw3a, m33.template triangularView<Eigen::Upper>()), make_dense_object_from<M33>(2, 4, 6, 0, 15, 18, 0, 0, 36)));

  // constant diagonal * triangular

  EXPECT_TRUE(is_near(contract(c31_2.asDiagonal(), m33.template triangularView<Eigen::Upper>()), make_dense_object_from<M33>(2, 4, 6, 0, 10, 12, 0, 0, 18)));
  static_assert(triangular_matrix<decltype(contract(c31_2.asDiagonal(), m33.template triangularView<Eigen::Upper>())), TriangleType::upper>);

  // triangular * triangular -- tested as part of adapters

  // hermitian * matrix

  EXPECT_TRUE(is_near(contract(m33.template selfadjointView<Eigen::Upper>(), m34), make_dense_object_from<M34>(74, 80, 86, 92, 159, 172, 185, 198, 222, 240, 258, 276)));
  EXPECT_TRUE(is_near(contract(m33.template selfadjointView<Eigen::Lower>(), m34), make_dense_object_from<M34>(156, 168, 180, 192, 203, 220, 237, 254, 272, 296, 320, 344)));

  // matrix * hermitian

  EXPECT_TRUE(is_near(contract(m23, m33.template selfadjointView<Eigen::Upper>()), make_dense_object_from<M23>(14, 30, 42, 32, 69, 96)));
  EXPECT_TRUE(is_near(contract(m23, m33.template selfadjointView<Eigen::Lower>()), make_dense_object_from<M23>(30, 38, 50, 66, 89, 122)));

  // hermitian * diagonal

  EXPECT_TRUE(is_near(contract(m33.template selfadjointView<Eigen::Upper>(), dm3b), make_dense_object_from<M33>(5, 12, 21, 10, 30, 42, 15, 36, 63)));

  // hermitian * constant diagonal

  EXPECT_TRUE(is_near(contract(m33.template selfadjointView<Eigen::Upper>(), c31_2.asDiagonal()), make_dense_object_from<M33>(2, 4, 6, 4, 10, 12, 6, 12, 18)));
  static_assert(hermitian_matrix<decltype(contract(m33.template selfadjointView<Eigen::Upper>(), c31_2.asDiagonal()))>);

  // diagonal * hermitian

  EXPECT_TRUE(is_near(contract(dw3a, m33.template selfadjointView<Eigen::Upper>()), make_dense_object_from<M33>(2, 4, 6, 6, 15, 18, 12, 24, 36)));

  // constant diagonal * hermitian

  EXPECT_TRUE(is_near(contract(c31_2.asDiagonal(), m33.template selfadjointView<Eigen::Upper>()), make_dense_object_from<M33>(2, 4, 6, 4, 10, 12, 6, 12, 18)));
  static_assert(hermitian_matrix<decltype(contract(c31_2.asDiagonal(), m33.template selfadjointView<Eigen::Upper>()))>);

  // hermitian * hermitian

  EXPECT_TRUE(is_near(contract(m33.template selfadjointView<Eigen::Upper>(), m33.template selfadjointView<Eigen::Upper>()),
    make_dense_object_from<M33>(14, 30, 42, 30, 65, 90, 42, 90, 126)));
  EXPECT_TRUE(is_near(contract(m33.template selfadjointView<Eigen::Lower>(), m33.template selfadjointView<Eigen::Lower>()),
    make_dense_object_from<M33>(66, 80, 102, 80, 105, 140, 102, 140, 194)));
}
