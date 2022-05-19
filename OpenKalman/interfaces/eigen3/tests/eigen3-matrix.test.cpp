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


TEST(eigen3, make_functions)
{
  auto m23 = M23 {};
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  EXPECT_TRUE(is_near(make_zero_matrix_like<M23>(Dimensions<2>{}, Dimensions<3>{}), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M00>(Dimensions<2>{}, 3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M00>(2, Dimensions<3>{}), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M00>(2, 3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m23), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m20_3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m03_2), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m00_23), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M23>(), M23::Zero()));

  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), Dimensions<3>()))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), 3))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<M00>(2, Dimensions<3>()))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<M00>(2, 3))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(m23))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(m20_3))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(m03_2))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(m00_23))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<M23>())>);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like<M00>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like<M00>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m20_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m03_2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like(m03_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m00_23)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like(m00_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M23>()), 0> == 2);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like<M00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like<M00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m20_3)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like(m20_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m03_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m00_23)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like(m00_23)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M23>()), 1> == 3);


  EXPECT_TRUE(is_near(make_constant_matrix_like<M23, 5>(Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, 5>(Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, 5>(2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, 5>(2, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<5>(m23), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<5>(m20_3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<5>(m03_2), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<5>(m00_23), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M23, 5>(), M23::Constant(5)));

  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(2, 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m23))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m20_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m03_2))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m00_23))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M23, 5>())> == 5);

  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<M00, 5>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<M00, 5>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m20_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m03_2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<5>(m03_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m00_23)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<5>(m00_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M23, 5>()), 0> == 2);

  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<M00, 5>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, 5>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<M00, 5>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m20_3)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<5>(m20_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m03_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<5>(m00_23)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<5>(m00_23)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M23, 5>()), 1> == 3);

  auto m22 = M22 {};
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

  EXPECT_TRUE(is_near(make_identity_matrix_like<M22>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<2>()), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(2), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(m22), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(m20_2), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(m02_2), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(m00_22), M22::Identity()));

  static_assert(identity_matrix<decltype(make_identity_matrix_like<M22>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<2>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(m22))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(m20_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(m02_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(m00_22))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M22>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M00>(Dimensions<2>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M00>(2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_identity_matrix_like<M00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m22)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m20_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m02_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m00_22)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_identity_matrix_like(m00_22)), 2);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M22>()), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M00>(Dimensions<2>())), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M00>(2)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_identity_matrix_like<M00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m22)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m20_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m02_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m00_22)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_identity_matrix_like(m00_22)), 2);
}


TEST(eigen3, get_index_dimension_of)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);

  static_assert(get_index_dimension_of<0>(m23) == 2);
  EXPECT_EQ(get_index_dimension_of<0>(M20 {m23}), 2);
  EXPECT_EQ(get_index_dimension_of<0>(M03 {m23}), 2);
  EXPECT_EQ(get_index_dimension_of<0>(M00 {m23}), 2);

  static_assert(get_index_dimension_of<1>(m23) == 3);
  EXPECT_EQ(get_index_dimension_of<1>(M20 {m23}), 3);
  EXPECT_EQ(get_index_dimension_of<1>(M03 {m23}), 3);
  EXPECT_EQ(get_index_dimension_of<1>(M00 {m23}), 3);
}


TEST(eigen3, get_dimensions_of)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);

  static_assert(dimension_size_of_v<decltype(get_dimensions_of<0>(m23))> == 2);
  static_assert(dimension_size_of_v<decltype(get_dimensions_of<0>(M20 {m23}))> == 2);
  EXPECT_EQ(get_dimension_size_of(get_dimensions_of<0>(M03 {m23})), 2);
  EXPECT_EQ(get_dimension_size_of(get_dimensions_of<0>(M00 {m23})), 2);

  static_assert(dimension_size_of_v<decltype(get_dimensions_of<1>(m23))> == 3);
  EXPECT_EQ(get_dimension_size_of(get_dimensions_of<1>(M20 {m23})), 3);
  static_assert(dimension_size_of_v<decltype(get_dimensions_of<1>(M03 {m23}))> == 3);
  EXPECT_EQ(get_dimension_size_of(get_dimensions_of<1>(M00 {m23})), 3);
}

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.
