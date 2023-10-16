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

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, get_index_count)
{
  static_assert(static_index_value_of_v<decltype(get_index_count(std::declval<M23>()))> == 2);
}


TEST(eigen3, get_vector_space_descriptor)
{
  M23 m23;

  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<0>(m23))> == 2);
  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<0>(M2x {m23}))> == 2);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<0>(Mx3 {m23})), 2);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<0>(Mxx {m23})), 2);

  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<1>(m23))> == 3);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<1>(M2x {m23})), 3);
  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<1>(Mx3 {m23}))> == 3);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<1>(Mxx {m23})), 3);
}


TEST(eigen3, get_index_dimension_of)
{
  M23 m23;

  static_assert(get_index_dimension_of<0>(m23) == 2);
  EXPECT_EQ(get_index_dimension_of<0>(M2x {m23}), 2);
  EXPECT_EQ(get_index_dimension_of<0>(Mx3 {m23}), 2);
  EXPECT_EQ(get_index_dimension_of<0>(Mxx {m23}), 2);

  static_assert(get_index_dimension_of<1>(m23) == 3);
  EXPECT_EQ(get_index_dimension_of<1>(M2x {m23}), 3);
  EXPECT_EQ(get_index_dimension_of<1>(Mx3 {m23}), 3);
  EXPECT_EQ(get_index_dimension_of<1>(Mxx {m23}), 3);
}


TEST(eigen3, get_tensor_order_of)
{
  M23 m23; static_assert(get_tensor_order_of(m23) == 2);
  EXPECT_EQ(get_tensor_order_of(M2x(2,3)), 2);
  EXPECT_EQ(get_tensor_order_of(Mx3(2,3)), 2);
  EXPECT_EQ(get_tensor_order_of(Mxx(2,3)), 2);

  M21 m21; static_assert(get_tensor_order_of(m21) == 1);
  EXPECT_EQ(get_tensor_order_of(M2x(2,1)), 1);
  EXPECT_EQ(get_tensor_order_of(Mx1(2,1)), 1);
  EXPECT_EQ(get_tensor_order_of(Mxx(2,1)), 1);

  M11 m11; static_assert(get_tensor_order_of(m11) == 0);
  EXPECT_EQ(get_tensor_order_of(M1x(1,1)), 0);
  EXPECT_EQ(get_tensor_order_of(Mx1(1,1)), 0);
  EXPECT_EQ(get_tensor_order_of(Mxx(1,1)), 0);
}


TEST(eigen3, get_all_dimensions_of)
{
  static_assert(dimension_size_of_v<decltype(std::get<0>(get_all_dimensions_of(std::declval<M23>())))> == 2);
  static_assert(dimension_size_of_v<decltype(std::get<0>(get_all_dimensions_of<M23>()))> == 2);
  static_assert(dimension_size_of_v<decltype(std::get<0>(get_all_dimensions_of(std::declval<M2x>())))> == 2);
  static_assert(dimension_size_of_v<decltype(std::get<1>(get_all_dimensions_of(std::declval<M23>())))> == 3);
  static_assert(dimension_size_of_v<decltype(std::get<1>(get_all_dimensions_of(std::declval<Mx3>())))> == 3);
  static_assert(std::tuple_size_v<decltype(get_all_dimensions_of(std::declval<Mxx>()))> == 2);

  EXPECT_EQ(std::get<0>(get_all_dimensions_of(Mxx(2, 3))), 2);
  EXPECT_EQ(std::get<1>(get_all_dimensions_of(Mxx(2, 3))), 3);
}


TEST(eigen3, get_vector_space_descriptor_match)
{
  EXPECT_TRUE(get_vector_space_descriptor_match(M23{}, Mxx(2, 3)));
  EXPECT_FALSE(get_vector_space_descriptor_match(M23{}, Mxx(2, 3), Mxx(2, 2)));
}


TEST(eigen3, get_is_square)
{
  EXPECT_TRUE(get_is_square(M22{}));
  EXPECT_FALSE(get_is_square(M23{}));
  EXPECT_TRUE(get_is_square(Mxx(3, 3)));
  EXPECT_FALSE(get_is_square(Mxx(2, 3)));
}


TEST(eigen3, get_is_one_by_one)
{
  EXPECT_TRUE(get_is_one_by_one(M11{}));
  EXPECT_FALSE(get_is_one_by_one(M12{}));
  EXPECT_TRUE(get_is_one_by_one(Mxx(1, 1)));
  EXPECT_FALSE(get_is_one_by_one(Mxx(2, 1)));
}


TEST(eigen3, get_is_vector)
{
  EXPECT_TRUE(get_is_vector(M11{}));
  EXPECT_TRUE(get_is_vector(M31{}));
  EXPECT_FALSE(get_is_vector(M32{}));
  EXPECT_FALSE(get_is_vector(M13{}));
  EXPECT_TRUE(get_is_vector(Mxx(1, 1)));
  EXPECT_TRUE(get_is_vector(Mxx(3, 1)));
  EXPECT_FALSE(get_is_vector(Mxx(3, 2)));
  EXPECT_FALSE(get_is_vector(Mxx(1, 3)));

  EXPECT_TRUE(get_is_vector<1>(M11{}));
  EXPECT_TRUE(get_is_vector<1>(M13{}));
  EXPECT_FALSE(get_is_vector<1>(M23{}));
  EXPECT_FALSE(get_is_vector<1>(M31{}));
  EXPECT_TRUE(get_is_vector<1>(Mxx(1, 1)));
  EXPECT_TRUE(get_is_vector<1>(Mxx(1, 3)));
  EXPECT_FALSE(get_is_vector<1>(Mxx(2, 3)));
  EXPECT_FALSE(get_is_vector<1>(Mxx(3, 1)));
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


TEST(eigen3, direct_access)
{
  Eigen::Matrix<double, 2, 2, Eigen::ColMajor> m22_1234c;
  m22_1234c << 1, 2, 3, 4;
  auto* datac = internal::raw_data(m22_1234c);
  static_assert(layout_of_v<decltype(m22_1234c)> == Layout::left);

  EXPECT_EQ(std::get<0>(internal::strides(m22_1234c)), 1);
  EXPECT_EQ(std::get<1>(internal::strides(m22_1234c)), 2);
  EXPECT_EQ(datac[0], 1);
  EXPECT_EQ(datac[1], 3);
  EXPECT_EQ(datac[2], 2);
  EXPECT_EQ(datac[3], 4);

  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> m22_1234r;
  m22_1234r << 1, 2, 3, 4;
  auto* datar = internal::raw_data(m22_1234r);
  static_assert(layout_of_v<decltype(m22_1234r)> == Layout::right);
  EXPECT_EQ(std::get<0>(internal::strides(m22_1234r)), 2);
  EXPECT_EQ(std::get<1>(internal::strides(m22_1234r)), 1);

  EXPECT_EQ(datar[0], 1);
  EXPECT_EQ(datar[1], 2);
  EXPECT_EQ(datar[2], 3);
  EXPECT_EQ(datar[3], 4);

  auto m22_1234r2 = make_dense_writable_matrix_from<M22, Layout::right, double>(1, 2, 3, 4);
  auto* datar2 = internal::raw_data(m22_1234r2);
  static_assert(layout_of_v<decltype(m22_1234r2)> == Layout::right);

  EXPECT_EQ(datar2[0], 1);
  EXPECT_EQ(datar2[1], 2);
  EXPECT_EQ(datar2[2], 3);
  EXPECT_EQ(datar2[3], 4);
}


TEST(eigen3, make_default_dense_writable_matrix_like)
{
  auto m23c = make_default_dense_writable_matrix_like<M11, Layout::left>(Dimensions<2>{}, Dimensions<3>{});
  static_assert(dimension_size_of_index_is<decltype(m23c), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(m23c), 1, 3>);
  static_assert(layout_of_v<decltype(m23c)> == Layout::left);
  EXPECT_EQ(std::get<0>(internal::strides(m23c)), 1);
  EXPECT_EQ(std::get<1>(internal::strides(m23c)), 2);

  auto m2xc_3 = make_default_dense_writable_matrix_like<M11, Layout::left>(Dimensions<2>{}, 3);
  static_assert(dimension_size_of_index_is<decltype(m2xc_3), 0, 2>);
  static_assert(dynamic_dimension<decltype(m2xc_3), 1>);
  static_assert(layout_of_v<decltype(m2xc_3)> == Layout::left);
  EXPECT_EQ(std::get<0>(internal::strides(m2xc_3)), 1);
  EXPECT_EQ(std::get<1>(internal::strides(m2xc_3)), 2);

  auto mx3c_2 = make_default_dense_writable_matrix_like<M11, Layout::left>(2, Dimensions<3>{});
  static_assert(dynamic_dimension<decltype(mx3c_2), 0>);
  static_assert(dimension_size_of_index_is<decltype(mx3c_2), 1, 3>);
  static_assert(layout_of_v<decltype(mx3c_2)> == Layout::left);
  EXPECT_EQ(std::get<0>(internal::strides(mx3c_2)), 1);
  EXPECT_EQ(std::get<1>(internal::strides(mx3c_2)), 2);

  auto mxxc_23 = make_default_dense_writable_matrix_like<M11, Layout::left>(2, 3);
  static_assert(dynamic_dimension<decltype(mxxc_23), 0>);
  static_assert(dynamic_dimension<decltype(mxxc_23), 1>);
  static_assert(layout_of_v<decltype(mxxc_23)> == Layout::left);
  EXPECT_EQ(std::get<0>(internal::strides(mxxc_23)), 1);
  EXPECT_EQ(std::get<1>(internal::strides(mxxc_23)), 2);

  auto m23r = make_default_dense_writable_matrix_like<M11, Layout::right>(Dimensions<2>{}, Dimensions<3>{});
  static_assert(dimension_size_of_index_is<decltype(m23r), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(m23r), 1, 3>);
  static_assert(layout_of_v<decltype(m23r)> == Layout::right);
  EXPECT_EQ(std::get<0>(internal::strides(m23r)), 3);
  EXPECT_EQ(std::get<1>(internal::strides(m23r)), 1);

  auto m2xr_3 = make_default_dense_writable_matrix_like<M11, Layout::right>(Dimensions<2>{}, 3);
  static_assert(dimension_size_of_index_is<decltype(m2xr_3), 0, 2>);
  static_assert(dynamic_dimension<decltype(m2xr_3), 1>);
  static_assert(layout_of_v<decltype(m2xr_3)> == Layout::right);
  EXPECT_EQ(std::get<0>(internal::strides(m2xr_3)), 3);
  EXPECT_EQ(std::get<1>(internal::strides(m2xr_3)), 1);

  auto mx3r_2 = make_default_dense_writable_matrix_like<M11, Layout::right>(2, Dimensions<3>{});
  static_assert(dynamic_dimension<decltype(mx3r_2), 0>);
  static_assert(dimension_size_of_index_is<decltype(mx3r_2), 1, 3>);
  static_assert(layout_of_v<decltype(mx3r_2)> == Layout::right);
  EXPECT_EQ(std::get<0>(internal::strides(mx3r_2)), 3);
  EXPECT_EQ(std::get<1>(internal::strides(mx3r_2)), 1);

  auto mxxr_23 = make_default_dense_writable_matrix_like<M11, Layout::right>(2, 3);
  static_assert(dynamic_dimension<decltype(mxxr_23), 0>);
  static_assert(dynamic_dimension<decltype(mxxr_23), 1>);
  static_assert(layout_of_v<decltype(mxxr_23)> == Layout::right);
  EXPECT_EQ(std::get<0>(internal::strides(mxxr_23)), 3);
  EXPECT_EQ(std::get<1>(internal::strides(mxxr_23)), 1);

  auto m11c = make_default_dense_writable_matrix_like<M11, Layout::left>(Dimensions<1>{}, Dimensions<1>{});
  static_assert(layout_of_v<decltype(m11c)> == Layout::left);
  auto mx1c_3 = make_default_dense_writable_matrix_like<M11, Layout::left>(3, Dimensions<1>{});
  static_assert(layout_of_v<decltype(mx1c_3)> == Layout::left);
  auto m1xc_3 = make_default_dense_writable_matrix_like<M11, Layout::right>(Dimensions<1>{}, 3);
  static_assert(layout_of_v<decltype(m1xc_3)> == Layout::right);
}


TEST(eigen3, make_dense_writable_matrix_from)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m32 = make_dense_writable_matrix_from<M32>(1, 2, 3, 4, 5, 6);
  auto cm22 = make_dense_writable_matrix_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M2x>(std::tuple{Dimensions<2>{}, 3}, 1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M2x>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M2x>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M2x>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<Mx2>(std::tuple{3, Dimensions<2>{}}, 1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<Mx2>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<Mx2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<Mx2>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<Mxx>(std::tuple{Dimensions<2>{}, 3}, 1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<Mxx>(std::tuple{2, 1}, 1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(index_dimension_of_v<decltype(make_dense_writable_matrix_from<M2x>(1, 2)), 1> == 1);
  static_assert(index_dimension_of_v<decltype(make_dense_writable_matrix_from<Mx2>(1, 2)), 0> == 1);
  static_assert(dynamic_dimension<decltype(make_dense_writable_matrix_from<Mxx>(std::tuple{2, 1}, 1, 2)), 0>);
  static_assert(index_dimension_of_v<decltype(make_dense_writable_matrix_from<Mxx>(std::tuple{Dimensions<2>{}, TypedIndex<>{}})), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_dense_writable_matrix_from<Mxx>(std::tuple{Dimensions<2>{}, TypedIndex<>{}})), 1> == 0);

  EXPECT_TRUE(is_near(make_eigen_matrix<double, 2, 2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix<double>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<cdouble, 2, 2>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(index_dimension_of_v<decltype(make_eigen_matrix<double>(1, 2)), 0> == 2);

  EXPECT_TRUE(is_near(make_eigen_matrix<2, 2>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_eigen_matrix(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix<2, 2>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(index_dimension_of_v<decltype(make_eigen_matrix(1., 2)), 0> == 2);
  static_assert(std::is_same_v<scalar_type_of_t<decltype(make_eigen_matrix(1., 2))>, double>);

  EXPECT_TRUE(is_near(make_eigen_matrix(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_eigen_matrix(1., 2, 3, 4), (eigen_matrix_t<double, 4, 1> {} << 1, 2, 3, 4).finished()));

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22), m22));

  EXPECT_TRUE(is_near(make_eigen_matrix<double, 1, 1>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_eigen_matrix<double>(4), eigen_matrix_t<double, 1, 1>(4)));

  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M2x {m22_93310};
  auto m02_93310 = Mx2 {m22_93310};
  auto m00_93310 = Mxx {m22_93310};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M2x {m22_3013};
  auto m02_3013 = Mx2 {m22_3013};
  auto m00_3013 = Mxx {m22_3013};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m20_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m02_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m00_3013.template triangularView<Eigen::Lower>()), m22_3013));

  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M2x {m22_3103};
  auto m02_3103 = Mx2 {m22_3103};
  auto m00_3103 = Mxx {m22_3103};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m22_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m20_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m02_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(m00_3103.template triangularView<Eigen::Upper>()), m22_3103));

  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mx1_2 = Mx1 {m21};
  auto mxx_21 = Mxx {m21};

  auto m22_1004 = make_dense_writable_matrix_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, 2> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {mxx_21}), m22_1004));

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));
}


TEST(eigen3, make_functions)
{
  auto m22h = make_dense_writable_matrix_from<M22>(3, 1, 1, 3);
  auto m22u = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m22l = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m22_uppert = Eigen::TriangularView<M22, Eigen::Upper> {m22h};
  auto m22_lowert = Eigen::TriangularView<M22, Eigen::Lower> {m22h};
  auto m22_upperh = Eigen::SelfAdjointView<M22, Eigen::Upper> {m22u};
  auto m22_lowerh = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22l};

  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::upper>(m22_uppert), m22u));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::upper>(m22_uppert))>);

  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::lower>(m22_lowert), m22l));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::lower>(m22_lowert))>);

  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::upper>(m22_upperh), m22u));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::upper>(m22_upperh))>);
  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::lower>(m22_upperh), m22l));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::lower>(m22_upperh))>);
  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::lower>(m22_lowerh), m22l));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::lower>(m22_lowerh))>);
  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::upper>(m22_lowerh), m22u));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::upper>(m22_lowerh))>);
}


  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.
