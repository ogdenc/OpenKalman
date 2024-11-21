/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, make_dense_object)
{
  auto m23c = make_dense_object<M11, Layout::left>(Dimensions<2>{}, Dimensions<3>{});
  static_assert(dimension_size_of_index_is<decltype(m23c), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(m23c), 1, 3>);
  static_assert(layout_of_v<decltype(m23c)> == Layout::left);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(m23c)), 1);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(m23c)), 2);

  auto m2xc_3 = make_dense_object<M11, Layout::left>(Dimensions<2>{}, 3);
  static_assert(dimension_size_of_index_is<decltype(m2xc_3), 0, 2>);
  static_assert(dynamic_dimension<decltype(m2xc_3), 1>);
  static_assert(layout_of_v<decltype(m2xc_3)> == Layout::left);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(m2xc_3)), 1);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(m2xc_3)), 2);

  auto mx3c_2 = make_dense_object<M11, Layout::left>(2, Dimensions<3>{});
  static_assert(dynamic_dimension<decltype(mx3c_2), 0>);
  static_assert(dimension_size_of_index_is<decltype(mx3c_2), 1, 3>);
  static_assert(layout_of_v<decltype(mx3c_2)> == Layout::left);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(mx3c_2)), 1);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(mx3c_2)), 2);

  auto mxxc_23 = make_dense_object<M11, Layout::left>(2, 3);
  static_assert(dynamic_dimension<decltype(mxxc_23), 0>);
  static_assert(dynamic_dimension<decltype(mxxc_23), 1>);
  static_assert(layout_of_v<decltype(mxxc_23)> == Layout::left);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(mxxc_23)), 1);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(mxxc_23)), 2);

  auto m23r = make_dense_object<M11, Layout::right>(Dimensions<2>{}, Dimensions<3>{});
  static_assert(dimension_size_of_index_is<decltype(m23r), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(m23r), 1, 3>);
  static_assert(layout_of_v<decltype(m23r)> == Layout::right);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(m23r)), 3);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(m23r)), 1);

  auto m2xr_3 = make_dense_object<M11, Layout::right>(Dimensions<2>{}, 3);
  static_assert(dimension_size_of_index_is<decltype(m2xr_3), 0, 2>);
  static_assert(dynamic_dimension<decltype(m2xr_3), 1>);
  static_assert(layout_of_v<decltype(m2xr_3)> == Layout::right);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(m2xr_3)), 3);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(m2xr_3)), 1);

  auto mx3r_2 = make_dense_object<M11, Layout::right>(2, Dimensions<3>{});
  static_assert(dynamic_dimension<decltype(mx3r_2), 0>);
  static_assert(dimension_size_of_index_is<decltype(mx3r_2), 1, 3>);
  static_assert(layout_of_v<decltype(mx3r_2)> == Layout::right);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(mx3r_2)), 3);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(mx3r_2)), 1);

  auto mxxr_23 = make_dense_object<M11, Layout::right>(2, 3);
  static_assert(dynamic_dimension<decltype(mxxr_23), 0>);
  static_assert(dynamic_dimension<decltype(mxxr_23), 1>);
  static_assert(layout_of_v<decltype(mxxr_23)> == Layout::right);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(mxxr_23)), 3);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(mxxr_23)), 1);

  auto m11c = make_dense_object<M11, Layout::left>(Dimensions<1>{}, Dimensions<1>{});
  static_assert(layout_of_v<decltype(m11c)> == Layout::left);
  auto mx1c_3 = make_dense_object<M11, Layout::left>(3, Dimensions<1>{});
  static_assert(layout_of_v<decltype(mx1c_3)> == Layout::left);
  auto m1xc_3 = make_dense_object<M11, Layout::right>(Dimensions<1>{}, 3);
  static_assert(layout_of_v<decltype(m1xc_3)> == Layout::right);

  auto m23y = to_dense_object<M11, Layout::left>(m23c);
  static_assert(dimension_size_of_index_is<decltype(m23y), 0, 2>);
  static_assert(dimension_size_of_index_is<decltype(m23y), 1, 3>);
  static_assert(layout_of_v<decltype(m23y)> == Layout::left);
  EXPECT_EQ(std::get<0>(OpenKalman::internal::strides(m23y)), 1);
  EXPECT_EQ(std::get<1>(OpenKalman::internal::strides(m23y)), 2);
}


TEST(eigen3, make_dense_object_from)
{
  auto m22 = make_dense_object_from<M22>(1, 2, 3, 4);
  auto m23 = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto m32 = make_dense_object_from<M32>(1, 2, 3, 4, 5, 6);
  auto cm22 = make_dense_object_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});

  EXPECT_TRUE(is_near(make_dense_object_from<M22>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(std::tuple{Dimensions<2>{}, 3}, 1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(std::tuple{3, Dimensions<2>{}}, 1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<Mxx>(std::tuple{Dimensions<2>{}, 3}, 1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_object_from<Mxx>(std::tuple{2, 1}, 1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<M2x>(1, 2)), 1> == 1);
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<Mx2>(1, 2)), 0> == 1);
  static_assert(dynamic_dimension<decltype(make_dense_object_from<Mxx>(std::tuple{2, 1}, 1, 2)), 0>);
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<Mxx>(std::tuple{Dimensions<2>{}, StaticDescriptor<>{}})), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<Mxx>(std::tuple{Dimensions<2>{}, StaticDescriptor<>{}})), 1> == 0);

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

  EXPECT_TRUE(is_near(to_dense_object(m22), m22));

  EXPECT_TRUE(is_near(make_eigen_matrix<double, 1, 1>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_eigen_matrix<double>(4), eigen_matrix_t<double, 1, 1>(4)));

  auto m22_93310 = make_dense_object_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M2x {m22_93310};
  auto m02_93310 = Mx2 {m22_93310};
  auto m00_93310 = Mxx {m22_93310};

  EXPECT_TRUE(is_near(to_dense_object(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(to_dense_object(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(to_dense_object(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(to_dense_object(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  auto m22_3013 = make_dense_object_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M2x {m22_3013};
  auto m02_3013 = Mx2 {m22_3013};
  auto m00_3013 = Mxx {m22_3013};

  EXPECT_TRUE(is_near(to_dense_object(m22_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(to_dense_object(m20_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(to_dense_object(m02_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(to_dense_object(m00_3013.template triangularView<Eigen::Lower>()), m22_3013));

  auto m22_3103 = make_dense_object_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M2x {m22_3103};
  auto m02_3103 = Mx2 {m22_3103};
  auto m00_3103 = Mxx {m22_3103};

  EXPECT_TRUE(is_near(to_dense_object(m22_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(to_dense_object(m20_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(to_dense_object(m02_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(to_dense_object(m00_3103.template triangularView<Eigen::Upper>()), m22_3103));

  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mx1_2 = Mx1 {m21};
  auto mxx_21 = Mxx {m21};

  auto m22_1004 = make_dense_object_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalMatrix<double, 2> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {mxx_21}), m22_1004));

  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));

  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  auto m44_1324 = make_dense_object_from<M44>(1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4);

  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<M22> {m22}), m44_1324));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<M2x> {m2x_2}), m44_1324));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<Mx2> {mx2_2}), m44_1324));
  EXPECT_TRUE(is_near(to_dense_object(Eigen::DiagonalWrapper<Mxx> {mxx_22}), m44_1324));

  static_assert(layout_of_v<decltype(make_dense_object_from<M22, Layout::left, double>(1, 2, 3, 4))> == Layout::left);
  static_assert(layout_of_v<decltype(make_dense_object_from<M22, Layout::right, double>(1, 2, 3, 4))> == Layout::right);
}


TEST(eigen3, make_special)
{
  auto m22h = make_dense_object_from<M22>(3, 1, 1, 3);
  auto m22u = make_dense_object_from<M22>(3, 1, 0, 3);
  auto m22l = make_dense_object_from<M22>(3, 0, 1, 3);
  auto m22_uppert = Eigen::TriangularView<M22, Eigen::Upper> {m22h};
  auto m22_lowert = Eigen::TriangularView<M22, Eigen::Lower> {m22h};
  auto m22_upperh = Eigen::SelfAdjointView<M22, Eigen::Upper> {m22u};
  auto m22_lowerh = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22l};

  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::upper>(m22h), m22u));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::upper>(m22h))>);
  EXPECT_TRUE(is_near(make_triangular_matrix<TriangleType::lower>(m22h), m22l));
  static_assert(eigen_TriangularView<decltype(make_triangular_matrix<TriangleType::lower>(m22h))>);

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

  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m22u), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22l))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m22l), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m22u))>);

  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m22_upperh), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_upperh))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m22_lowerh), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m22_lowerh))>);

  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m22_uppert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_uppert))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m22_uppert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_uppert))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m22_lowert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_lowert))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m22_lowert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m22_lowert))>);

  EXPECT_TRUE(is_near(make_identity_matrix_like(M33{}), M33::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like(M33{}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like(M34{}), M34::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like(M34{}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like(M43{}), M43::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like(M43{}))>);
  static_assert(std::is_same_v<scalar_type_of_t<decltype(make_identity_matrix_like<int>(M43{}))>, int>);

  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions<3>{}), M33::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions<3>{}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions<4>{}), M34::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions<4>{}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<4>{}, Dimensions<3>{}), M43::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<4>{}, Dimensions<3>{}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions{3}), M3x::Identity(3, 3)));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions{3}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions{4}), M3x::Identity(3, 4)));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions{4}))>);
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions{3}, Dimensions{4}), Mxx::Identity(3, 4)));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions{3}, Dimensions{4}))>);
}

