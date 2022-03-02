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


TEST(eigen3, nested_matrix)
{
  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m21 = M21 {1, 4};

  EXPECT_TRUE(is_near(nested_matrix(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}), m22_93310));
  EXPECT_TRUE(is_near(nested_matrix(Eigen::TriangularView<M22, Eigen::Upper> {m22_3103}), m22_3103));
  EXPECT_TRUE(is_near(nested_matrix(Eigen::DiagonalMatrix<double, 2> {m21}), m21));
  EXPECT_TRUE(is_near(nested_matrix(Eigen::DiagonalWrapper<M21> {m21}), m21));
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


TEST(eigen3, make_dense_writable_matrix_from)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m32 = make_dense_writable_matrix_from<M32>(1, 2, 3, 4, 5, 6);
  auto cm22 = make_dense_writable_matrix_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M20>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M20>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M20>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M02>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M02>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M02>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M00>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(column_dimension_of_v<decltype(make_dense_writable_matrix_from<M20>(1, 2))> == 1);
  static_assert(row_dimension_of_v<decltype(make_dense_writable_matrix_from<M02>(1, 2))> == 1);
  static_assert(row_dimension_of_v<decltype(make_dense_writable_matrix_from<M00>(1, 2))> == 2);

  EXPECT_TRUE(is_near(make_eigen_matrix<double, 2, 2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, 2, dynamic_size>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, 2, dynamic_size>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, 2, dynamic_size>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, dynamic_size, 2>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, dynamic_size, 2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, dynamic_size, 2>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, dynamic_size, dynamic_size>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<cdouble, 2, 2>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(column_dimension_of_v<decltype(make_eigen_matrix<double, 2, dynamic_size>(1, 2))> == 1);
  static_assert(row_dimension_of_v<decltype(make_eigen_matrix<double, dynamic_size, 2>(1, 2))> == 1);
  static_assert(row_dimension_of_v<decltype(make_eigen_matrix<double, dynamic_size, dynamic_size>(1, 2))> == 2);

  EXPECT_TRUE(is_near(make_eigen_matrix<2, 2>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<2, dynamic_size>(1., 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_eigen_matrix<2, dynamic_size>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<2, dynamic_size>(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<dynamic_size, 2>(1., 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_eigen_matrix<dynamic_size, 2>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<dynamic_size, 2>(1., 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<dynamic_size, dynamic_size>(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<2, 2>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(column_dimension_of_v<decltype(make_eigen_matrix<2, dynamic_size>(1., 2))> == 1);
  static_assert(row_dimension_of_v<decltype(make_eigen_matrix<dynamic_size, 2>(1., 2))> == 1);
  static_assert(row_dimension_of_v<decltype(make_eigen_matrix<dynamic_size, dynamic_size>(1., 2))> == 2);

  EXPECT_TRUE(is_near(make_eigen_matrix(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix(1., 2, 3, 4), (eigen_matrix_t<double, 4, 1> {} << 1, 2, 3, 4).finished()));

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22), m22));

  EXPECT_TRUE(is_near(make_eigen_matrix<double, 1, 1>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, 1, dynamic_size>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, dynamic_size, 1>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_eigen_matrix<double, dynamic_size, dynamic_size>(4), eigen_matrix_t<double, 1, 1>(4)));

  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M20 {m22_93310};
  auto m02_93310 = M02 {m22_93310};
  auto m00_93310 = M00 {m22_93310};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M20 {m22_3013};
  auto m02_3013 = M02 {m22_3013};
  auto m00_3013 = M00 {m22_3013};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m20_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m02_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m00_3013.template triangularView<Eigen::Lower>()), m22_3013));

  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M20 {m22_3103};
  auto m02_3103 = M02 {m22_3103};
  auto m00_3103 = M00 {m22_3103};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m20_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m02_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m00_3103.template triangularView<Eigen::Upper>()), m22_3103));

  auto m21 = M21 {1, 4};
  auto m20_1 = M20 {m21};
  auto m01_2 = M01 {m21};
  auto m00_21 = M00 {m21};

  auto m22_1004 = make_dense_writable_matrix_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, 2> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m00_21}), m22_1004));

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<M20> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<M01> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<M00> {m00_21}), m22_1004));
}


TEST(eigen3, runtime_dimension_of)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);

  EXPECT_EQ(runtime_dimension_of<0>(m23), 2);
  EXPECT_EQ(runtime_dimension_of<0>(M20 {m23}), 2);
  EXPECT_EQ(runtime_dimension_of<0>(M03 {m23}), 2);
  EXPECT_EQ(runtime_dimension_of<0>(M00 {m23}), 2);

  EXPECT_EQ(runtime_dimension_of<1>(m23), 3);
  EXPECT_EQ(runtime_dimension_of<1>(M20 {m23}), 3);
  EXPECT_EQ(runtime_dimension_of<1>(M03 {m23}), 3);
  EXPECT_EQ(runtime_dimension_of<1>(M00 {m23}), 3);
}

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.


TEST(eigen3, column)
{
  auto m33 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto c2 = make_dense_writable_matrix_from<M31>(0, 0, 3);

  EXPECT_TRUE(is_near(column(m33, 2), c2));
  EXPECT_TRUE(is_near(column(M30 {m33}, 2), c2));
  EXPECT_TRUE(is_near(column(M03 {m33}, 2), c2));
  EXPECT_TRUE(is_near(column(M00 {m33}, 2), c2));

  EXPECT_TRUE(is_near(column(m33.array(), 2), c2));
  EXPECT_TRUE(is_near(column(M30 {m33}.array(), 2), c2));
  EXPECT_TRUE(is_near(column(M03 {m33}.array(), 2), c2));
  EXPECT_TRUE(is_near(column(M00 {m33}.array(), 2), c2));

  static_assert(column_vector<decltype(column(M00 {m33}, 2))>);
  static_assert(column_vector<decltype(column(M00 {m33}.array(), 2))>);

  auto c1 = make_dense_writable_matrix_from<M31>(0, 2, 0);

  EXPECT_TRUE(is_near(column<1>(m33), c1));
  EXPECT_TRUE(is_near(column(M30 {m33}, 1), c1));
  EXPECT_TRUE(is_near(column(M03 {m33}, 1), c1));
  EXPECT_TRUE(is_near(column(M00 {m33}, 1), c1));

  EXPECT_TRUE(is_near(column<1>(m33.array()), c1));
  EXPECT_TRUE(is_near(column(M30 {m33}.array(), 1), c1));
  EXPECT_TRUE(is_near(column(M03 {m33}.array(), 1), c1));
  EXPECT_TRUE(is_near(column(M00 {m33}.array(), 1), c1));

  static_assert(column_vector<decltype(column(M00 {m33}, 1))>);
  static_assert(column_vector<decltype(column(M00 {m33}.array(), 1))>);
}


TEST(eigen3, row)
{
  auto m33 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto r2 = make_dense_writable_matrix_from<M13>(0, 0, 3);

  EXPECT_TRUE(is_near(row(m33, 2), r2));
  EXPECT_TRUE(is_near(row(M30 {m33}, 2), r2));
  EXPECT_TRUE(is_near(row(M03 {m33}, 2), r2));
  EXPECT_TRUE(is_near(row(M00 {m33}, 2), r2));

  EXPECT_TRUE(is_near(row(m33.array(), 2), r2));
  EXPECT_TRUE(is_near(row(M30 {m33}.array(), 2), r2));
  EXPECT_TRUE(is_near(row(M03 {m33}.array(), 2), r2));
  EXPECT_TRUE(is_near(row(M00 {m33}.array(), 2), r2));

  static_assert(row_vector<decltype(row(M00 {m33}, 2))>);
  static_assert(row_vector<decltype(row(M00 {m33}.array(), 2))>);

  auto r1 = make_dense_writable_matrix_from<M13>(0, 2, 0);

  EXPECT_TRUE(is_near(row<1>(m33), r1));
  EXPECT_TRUE(is_near(row(M30 {m33}, 1), r1));
  EXPECT_TRUE(is_near(row(M03 {m33}, 1), r1));
  EXPECT_TRUE(is_near(row(M00 {m33}, 1), r1));

  EXPECT_TRUE(is_near(row<1>(m33.array()), r1));
  EXPECT_TRUE(is_near(row(M30 {m33}.array(), 1), r1));
  EXPECT_TRUE(is_near(row(M03 {m33}.array(), 1), r1));
  EXPECT_TRUE(is_near(row(M00 {m33}.array(), 1), r1));

  static_assert(row_vector<decltype(row(M00 {m33}, 1))>);
  static_assert(row_vector<decltype(row(M00 {m33}.array(), 1))>);
 }

