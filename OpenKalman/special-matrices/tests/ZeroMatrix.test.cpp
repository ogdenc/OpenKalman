/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "special-matrices.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using std::numbers::pi;

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
  using M40 = eigen_matrix_t<double, 4, dynamic_size>;
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using Axis2 = TypedIndex<Axis, Axis>;

  /*auto z11 = M11::Identity() - M11::Identity(); using Z11 = decltype(z11);
  auto z22 = M22::Identity() - M22::Identity(); using Z22 = decltype(z22);
  auto z21 = z22.diagonal(); using Z21 = decltype(z21);
  auto z12 = z11.replicate<1,2>(); using Z12 = decltype(z12);
  auto z23 = z11.replicate<2,3>(); using Z23 = decltype(z23);
  using Z20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic>; 
  using Z02 = Eigen::Replicate<Z11, Eigen::Dynamic, 2>; 
  using Z00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic>; 
  using Z01 = Eigen::Replicate<Z11, Eigen::Dynamic, 1>; 
  auto z20_2 = Z20 {z11, 2, 2};
  auto z02_2 = Z02 {z11, 2, 2};
  auto z00_22 = Z00 {z11, 2, 2};
  auto z01_2 = Z01 {z11, 2, 1};
  auto z20_1 = Z20 {z11, 2, 1};
  auto z00_21 = Z00 {z11, 2, 1};

  auto c11_1 = M11::Identity(); using C11_1 = decltype(c11_1);
  auto c11_m1 = -M11::Identity(); using C11_m1 = decltype(c11_m1);
  auto c11_2 = c11_1 + c11_1; using C11_2 = decltype(c11_2);
  auto c11_m2 = -c11_2; using C11_m2 = decltype(c11_m2);
  auto c21_1 = c11_1.replicate<2,1>(); using C21_1 = decltype(c21_1);
  auto c21_2 = c11_2.replicate<2,1>(); using C21_2 = decltype(c21_2);
  auto c21_m2 = c11_m2.replicate<2,1>(); using C21_m2 = decltype(c21_m2);
  auto c12_2 = c11_2.replicate<1,2>(); using C12_2 = decltype(c12_2);
  auto c22_2 = c11_2.replicate<2,2>(); using C22_2 = decltype(c22_2);
  auto c22_m2 = c11_m2.replicate<2,2>(); using C22_m2 = decltype(c22_m2);
  using C20_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using C02_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C00_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;
  using C01_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  auto c20_2_2 = C20_2 {c11_2, 2, 2};
  auto c02_2_2 = C02_2 {c11_2, 2, 2};
  auto c00_22_2 = C00_2 {c11_2, 2, 2};
  auto c20_1_2 = C20_2 {c11_2, 2, 1};
  auto c01_2_2 = C01_2 {c11_2, 2, 1};
  auto c00_21_2 = C00_2 {c11_2, 2, 1};

  auto b22_true = eigen_matrix_t<bool, 1, 1>::Identity().replicate<2,2>(); using B22_true = decltype(b22_true);*/
}


TEST(eigen3, ZeroMatrix_traits)
{
  static_assert(eigen_zero_expr<ZeroMatrix<M31>>);
  static_assert(zero_matrix<ZeroMatrix<M31>>);

  static_assert(eigen_matrix<ZeroMatrix<M31>>);

  static_assert(diagonal_matrix<ZeroMatrix<M33>>);
  static_assert(not diagonal_matrix<ZeroMatrix<M31>>);
  static_assert(not diagonal_matrix<ZeroMatrix<M00>>);

  static_assert(self_adjoint_matrix<ZeroMatrix<M33>>);
  static_assert(self_adjoint_matrix<ZeroMatrix<eigen_matrix_t<cdouble, 3, 3>>>);
  static_assert(not self_adjoint_matrix<ZeroMatrix<M31>>);
  static_assert(not self_adjoint_matrix<ZeroMatrix<M00>>);

  static_assert(upper_triangular_matrix<ZeroMatrix<M33>>);
  static_assert(not upper_triangular_matrix<ZeroMatrix<M31>>);
  static_assert(not upper_triangular_matrix<ZeroMatrix<M00>>);

  static_assert(lower_triangular_matrix<ZeroMatrix<M33>>);
  static_assert(not lower_triangular_matrix<ZeroMatrix<M31>>);
  static_assert(not lower_triangular_matrix<ZeroMatrix<M00>>);

  static_assert(square_matrix<ZeroMatrix<M33>>);
  static_assert(not square_matrix<ZeroMatrix<M31>>);
  static_assert(not square_matrix<ZeroMatrix<M00>>);

  static_assert(one_by_one_matrix<ZeroMatrix<M11>>);
  static_assert(not one_by_one_matrix<ZeroMatrix<M10>>);
  static_assert(not one_by_one_matrix<ZeroMatrix<M00>>);

  static_assert(element_gettable<ZeroMatrix<M22>, std::size_t, std::size_t>);
  static_assert(element_gettable<ZeroMatrix<M20>, std::size_t, std::size_t>);
  static_assert(element_gettable<ZeroMatrix<M02>, std::size_t, std::size_t>);
  static_assert(element_gettable<ZeroMatrix<M00>, std::size_t, std::size_t>);

  static_assert(not element_settable<ZeroMatrix<M22>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ZeroMatrix<M20>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ZeroMatrix<M02>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ZeroMatrix<M00>&, std::size_t, std::size_t>);

  static_assert(dynamic_rows<ZeroMatrix<M00>>);
  static_assert(dynamic_rows<ZeroMatrix<M02>>);
  static_assert(not dynamic_rows<ZeroMatrix<M20>>);

  static_assert(dynamic_columns<ZeroMatrix<M00>>);
  static_assert(not dynamic_columns<ZeroMatrix<M02>>);
  static_assert(dynamic_columns<ZeroMatrix<M20>>);

  static_assert(not writable<ZeroMatrix<M33>>);
  static_assert(modifiable<M33, ZeroMatrix<M33>>);
}


TEST(eigen3, ZeroMatrix_class)
{
  ZeroMatrix<M23> z23;
  ZeroMatrix<M20> z20 {3};
  ZeroMatrix<M03> z03 {2};
  ZeroMatrix<M00> z00 {2, 3};

  EXPECT_TRUE(is_near(z23, M23::Zero()));
  EXPECT_TRUE(is_near(z20, M23::Zero()));
  EXPECT_TRUE(is_near(z03, M23::Zero()));
  EXPECT_TRUE(is_near(z00, M23::Zero()));

  EXPECT_TRUE(is_near(ZeroMatrix {z23}, M23::Zero()));
  EXPECT_TRUE(is_near(ZeroMatrix {z20}, M23::Zero()));
  EXPECT_TRUE(is_near(ZeroMatrix {z03}, M23::Zero()));
  EXPECT_TRUE(is_near(ZeroMatrix {z00}, M23::Zero()));

  EXPECT_TRUE(is_near(ZeroMatrix {ZeroMatrix<M23> {}}, M23::Zero()));
  EXPECT_TRUE(is_near(ZeroMatrix {ZeroMatrix<M20> {3}}, M23::Zero()));
  EXPECT_TRUE(is_near(ZeroMatrix {ZeroMatrix<M03> {2}}, M23::Zero()));
  EXPECT_TRUE(is_near(ZeroMatrix {ZeroMatrix<M00> {2,3}}, M23::Zero()));

  EXPECT_NEAR((ZeroMatrix {ConstantMatrix<M23, 0>{}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix {ConstantMatrix<M20, 0>{3}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix {ConstantMatrix<M03, 0>{2}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix {ConstantMatrix<M00, 0>{2,3}}(1, 2)), 0, 1e-6);

  auto nz11 = M11::Identity() - M11::Identity(); using Z11 = decltype(nz11);
  auto nz23 = nz11.replicate<2,3>();
  auto nz20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {nz11, 2, 3};
  auto nz03 = Eigen::Replicate<Z11, Eigen::Dynamic, 3> {nz11, 2, 3};
  auto nz00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {nz11, 2, 3};

  z20 = nz23; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = nz23; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = nz23; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = nz20; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z03 = nz20; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = nz20; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = nz03; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = nz03; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z00 = nz03; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = nz00; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = nz00; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = nz00; EXPECT_TRUE(is_near(z03, M23::Zero()));

  z23 = ZeroMatrix {nz23}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ZeroMatrix {nz23}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ZeroMatrix {nz23}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ZeroMatrix {nz23}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = ZeroMatrix {nz20}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ZeroMatrix {nz20}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ZeroMatrix {nz20}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ZeroMatrix {nz20}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = ZeroMatrix {nz03}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ZeroMatrix {nz03}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ZeroMatrix {nz03}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ZeroMatrix {nz03}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = ZeroMatrix {nz00}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ZeroMatrix {nz00}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ZeroMatrix {nz00}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ZeroMatrix {nz00}; EXPECT_TRUE(is_near(z00, M23::Zero()));

  EXPECT_NEAR((ZeroMatrix<M22> {}(0, 0)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M22> {}(0, 1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M20> {2}(0, 1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M02> {2}(0, 1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M00> {2,2}(0, 1)), 0, 1e-6);

  EXPECT_NEAR((ZeroMatrix<M31> {}(1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M01> {3}(1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M13> {}(1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M10> {3}(1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M31> {}[1]), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M01> {3}[1]), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M13> {}[1]), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<M10> {3}[1]), 0, 1e-6);
}


TEST(eigen3, ZeroMatrix_functions)
{
  ZeroMatrix<M23> z23 {Dimensions<2>(), Dimensions<3>()};
  ZeroMatrix<M20> z20_3 {Dimensions<2>(), 3};
  ZeroMatrix<M03> z03_2 {2, Dimensions<3>()};
  ZeroMatrix<M00> z00_23 {2, 3};

  using Z23 = decltype(z23);
  using Z00 = decltype(z00_23);

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z23), M23::Zero()));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z20_3), M23::Zero()));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z03_2), M23::Zero()));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z00_23), M23::Zero()));

  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<Z23>())>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<Z00>(Dimensions<2>(), Dimensions<3>()))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<Z00>(Dimensions<2>(), 3))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<Z00>(2, Dimensions<3>()))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like<Z00>(2, 3))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(z23))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(z20_3))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(z03_2))>);
  static_assert(eigen_zero_expr<decltype(make_zero_matrix_like(z00_23))>);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z23>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<Z00>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<Z00>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z20_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z03_2)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like(z03_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z00_23)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like(z00_23)), 2);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z23>()), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<Z00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<Z00>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<Z00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z20_3)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like(z20_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z03_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z00_23)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like(z00_23)), 3);

  ZeroMatrix<M22> z22 {Dimensions<2>(), Dimensions<2>()};
  ZeroMatrix<M20> z20_2 {Dimensions<2>(), 2};
  ZeroMatrix<M02> z02_2 {2, Dimensions<2>()};
  ZeroMatrix<M00> z00_22 {2, 2};

  using Z22 = decltype(z22);
  using Z02 = decltype(z02_2);

  static_assert(identity_matrix<decltype(make_identity_matrix_like<Z22>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<Z00>(Dimensions<2>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<Z00>(2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z22))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z20_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z02_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z00_22))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Z22>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Z00>(Dimensions<2>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Z00>(2)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like<Z00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z22)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z20_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z02_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z00_22)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like(z00_22)), 2);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Z22>()), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Z00>(Dimensions<2>())), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Z00>(2)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like<Z00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z22)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z20_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z02_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z00_22)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like(z00_22)), 2);

  EXPECT_TRUE(is_near(make_self_contained(ZeroMatrix {z23}), M23::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ZeroMatrix {z20_3}), M23::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ZeroMatrix {z03_2}), M23::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ZeroMatrix {z00_23}), M23::Zero()));

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  ZeroMatrix<M21> z21;
  ZeroMatrix<M20> z20_1 {1};
  ZeroMatrix<M01> z01_2 {2};
  ZeroMatrix<M00> z00_21 {2, 1};

  ZeroMatrix<M12> z12;
  ZeroMatrix<M10> z10_2 {2};
  ZeroMatrix<M02> z02_1 {1};
  ZeroMatrix<M00> z00_12 {1, 2};

  EXPECT_TRUE(is_near(to_diagonal(z21), z22));
  EXPECT_TRUE(is_near(to_diagonal(z20_1), z22));
  EXPECT_TRUE(is_near(to_diagonal(z01_2), z22));
  EXPECT_TRUE(is_near(to_diagonal(z00_21), z22));
  static_assert(eigen_zero_expr<decltype(to_diagonal(z20_1))>);
  static_assert(zero_matrix<decltype(to_diagonal(z00_21))>);
  static_assert(diagonal_matrix<decltype(to_diagonal(z00_21))>);

  EXPECT_TRUE(is_near(diagonal_of(z22), z21));
  EXPECT_TRUE(is_near(diagonal_of(z20_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(z02_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(z00_22), z21));
  static_assert(eigen_zero_expr<decltype(diagonal_of(z00_22))>);

  EXPECT_TRUE(is_near(transpose(z23), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z20_3), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z03_2), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z00_23), M32::Zero()));
  static_assert(eigen_zero_expr<decltype(transpose(z00_23))>);

  EXPECT_TRUE(is_near(adjoint(z23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z20_3), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z03_2), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z00_23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(ZeroMatrix<eigen_matrix_t<cdouble, 2, 3>> {}), M32::Zero()));
  static_assert(eigen_zero_expr<decltype(adjoint(z00_23))>);

  EXPECT_NEAR(determinant(z22), 0, 1e-6);
  EXPECT_NEAR(determinant(z20_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z02_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z00_22), 0, 1e-6);

  EXPECT_NEAR(trace(z22), 0, 1e-6);
  EXPECT_NEAR(trace(z20_2), 0, 1e-6);
  EXPECT_NEAR(trace(z02_2), 0, 1e-6);
  EXPECT_NEAR(trace(z00_22), 0, 1e-6);

  auto m1234 = make_eigen_matrix<double, 2, 2>(1, 2, 3, 4);

  EXPECT_TRUE(is_near(rank_update(z22, m1234, 0.25), 0.5*m1234));
  EXPECT_TRUE(is_near(rank_update(z20_2, m1234, 0.25), 0.5*m1234));
  EXPECT_TRUE(is_near(rank_update(z02_2, m1234, 0.25), 0.5*m1234));
  EXPECT_TRUE(is_near(rank_update(z00_22, m1234, 0.25), 0.5*m1234));

  auto di5 = M22::Identity() * 5;

  EXPECT_TRUE(is_near(rank_update(z22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update(z20_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update(z02_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update(z00_22, di5, 0.25), 0.5*di5));

  EXPECT_TRUE(is_near(solve(z22, z23), z23));
  EXPECT_TRUE(is_near(solve(z22, z00_23), z23));
  EXPECT_TRUE(is_near(solve(z22, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z22, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z23), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z00_23), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z23), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z00_23), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z23), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z00_23), z23));

  EXPECT_TRUE(is_near(average_reduce<1>(z22), z21));
  EXPECT_TRUE(is_near(average_reduce<1>(z20_2), z21));
  EXPECT_TRUE(is_near(average_reduce<1>(z02_2), z21));
  EXPECT_TRUE(is_near(average_reduce<1>(z00_22), z21));
  static_assert(zero_matrix<decltype(average_reduce<1>(z00_22))>);

  EXPECT_TRUE(is_near(average_reduce<0>(z22), z12));
  EXPECT_TRUE(is_near(average_reduce<0>(z20_2), z12));
  EXPECT_TRUE(is_near(average_reduce<0>(z02_2), z12));
  EXPECT_TRUE(is_near(average_reduce<0>(z00_22), z12));
  static_assert(zero_matrix<decltype(average_reduce<0>(z00_22))>);

  EXPECT_TRUE(is_near(LQ_decomposition(z23), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z20_3), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z03_2), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z00_23), z22));
  static_assert(zero_matrix<decltype(LQ_decomposition(z00_23))>);

  ZeroMatrix<M32> z32;
  ZeroMatrix<M30> z30_2 {2};
  ZeroMatrix<M02> z02_3 {3};
  ZeroMatrix<M00> z00_32 {3, 2};

  EXPECT_TRUE(is_near(QR_decomposition(z32), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z30_2), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z02_3), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z00_32), z22));
  static_assert(zero_matrix<decltype(QR_decomposition(z00_32))>);

  auto tup_z33_z23 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M23>()};

  EXPECT_TRUE(is_near(split_vertical<3, 2>(make_zero_matrix_like<M00>(Dimensions<5>{}, Dimensions<3>{})), tup_z33_z23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(make_zero_matrix_like<M00>(Dimensions<5>{}, 3)), tup_z33_z23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(make_zero_matrix_like<M00>(5, Dimensions<3>{})), tup_z33_z23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(make_zero_matrix_like<M00>(5, 3)), tup_z33_z23));

  auto tup_z33_z32 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M32>()};

  EXPECT_TRUE(is_near(split_horizontal<3, 2>(make_zero_matrix_like<M00>(Dimensions<3>{}, Dimensions<5>{})), tup_z33_z32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(make_zero_matrix_like<M00>(Dimensions<3>{}, 5)), tup_z33_z32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(make_zero_matrix_like<M00>(3, Dimensions<5>{})), tup_z33_z32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(make_zero_matrix_like<M00>(3, 5)), tup_z33_z32));

  auto tup_z33_z22 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M22>()};

  EXPECT_TRUE(is_near(split_diagonal<3, 2>(make_zero_matrix_like<M55>(Dimensions<5>{}, Dimensions<5>{})), tup_z33_z22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(make_zero_matrix_like<M50>(Dimensions<5>{}, 5)), tup_z33_z22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(make_zero_matrix_like<M05>(5, Dimensions<5>{})), tup_z33_z22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(make_zero_matrix_like<M00>(5, 5)), tup_z33_z22));

  EXPECT_NEAR(get_element(z22, 1, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z20_2, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z02_2, 0, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_22, 1, 0), 0, 1e-6);

  EXPECT_NEAR(get_element(z21, 0, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z20_1, 1, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z01_2, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_21, 1, 0), 0, 1e-6);

  EXPECT_NEAR(get_element(z21, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z20_1, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z01_2, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_21, 1), 0, 1e-6);

  EXPECT_NEAR(get_element(z12, 0, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z10_2, 0, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z02_1, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_12, 0, 1), 0, 1e-6);

  EXPECT_NEAR(get_element(z12, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z10_2, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z02_1, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_12, 1), 0, 1e-6);

  auto zc34 = ZeroMatrix<M34> {};

  EXPECT_TRUE(is_near(column<1>(ZeroMatrix<M23>()), (M21::Zero())));
  EXPECT_TRUE(is_near(column(ZeroMatrix<M23>(), 1), (M21::Zero())));
  auto czc34 = column(zc34, 1);
  EXPECT_EQ(get_dimensions_of<0>(czc34), 3);
  static_assert(get_dimensions_of<1>(czc34) == 1);
  static_assert(zero_matrix<decltype(czc34)>);
  auto czv34 = column<1>(ZeroMatrix<M04> {3});
  EXPECT_EQ(get_dimensions_of<0>(czv34), 3);
  static_assert(get_dimensions_of<1>(czv34) == 1);
  static_assert(zero_matrix<decltype(czv34)>);

  EXPECT_TRUE(is_near(column<1>(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(column(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(column(make_zero_matrix_like<M30>(Dimensions<3>{}, 3), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(column(make_zero_matrix_like<M03>(3, Dimensions<3>{}), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(column(make_zero_matrix_like<M00>(3,3), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));

  EXPECT_TRUE(is_near(row<1>(ZeroMatrix<M32>(Dimensions<3>{}, Dimensions<2>{})), (M12::Zero())));
  EXPECT_TRUE(is_near(row(ZeroMatrix<M32>(Dimensions<3>{}, Dimensions<2>{}), 1), (M12::Zero())));
  auto rzc34 = row(zc34, 1);
  EXPECT_EQ(get_dimensions_of<1>(rzc34), 4);
  static_assert(get_dimensions_of<0>(rzc34) == 1);
  static_assert(zero_matrix<decltype(rzc34)>);
  auto rzv34 = row<1>(ZeroMatrix<M30> {Dimensions<3>{}, 4});
  EXPECT_EQ(get_dimensions_of<1>(rzv34), 4);
  static_assert(get_dimensions_of<0>(rzv34) == 1);
  static_assert(zero_matrix<decltype(rzv34)>);

  EXPECT_TRUE(is_near(row<1>(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), make_dense_writable_matrix_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(row(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(row(make_zero_matrix_like<M30>(Dimensions<3>{}, 3), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(row(make_zero_matrix_like<M03>(3, Dimensions<3>{}), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(row(make_zero_matrix_like<M00>(3,3), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));
}


TEST(eigen3, ZeroMatrix_arithmetic)
{
  auto m22y = make_eigen_matrix<double, 2, 2>(1, 2, 3, 4);
  EXPECT_TRUE(is_near(z00 + m22y, m22y, 1e-6));
  EXPECT_TRUE(is_near(m22y + z00, m22y, 1e-6));
  static_assert(zero_matrix<decltype(z00 + z00)>);
  EXPECT_TRUE(is_near(m22y - z00, m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y, -m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y.Identity(), -m22y.Identity(), 1e-6));
  static_assert(diagonal_matrix<decltype(z00 - m22y.Identity())>);
  static_assert(zero_matrix<decltype(z00 - z00)>);
  EXPECT_TRUE(is_near(z00 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * m22y, z00, 1e-6));
  EXPECT_TRUE(is_near(m22y * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * 2, z00, 1e-6));
  EXPECT_TRUE(is_near(2 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 / 2, z00, 1e-6));
  EXPECT_TRUE(is_near(-z00, z00, 1e-6));

  EXPECT_EQ((ZeroMatrix<M43>::rows()), 4);
  EXPECT_EQ((ZeroMatrix<M43>::cols()), 3);
  EXPECT_TRUE(is_near(make_identity_matrix_like<ZeroMatrix<M22>>(), M22::Identity()));
}

