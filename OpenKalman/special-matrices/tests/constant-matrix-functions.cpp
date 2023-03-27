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

using numbers::pi;

namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M14 = eigen_matrix_t<double, 1, 4>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M24 = eigen_matrix_t<double, 2, 4>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M42 = eigen_matrix_t<double, 4, 2>;
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

  using CM11 = eigen_matrix_t<cdouble, 1, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM33 = eigen_matrix_t<cdouble, 3, 3>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using Axis2 = TypedIndex<Axis, Axis>;

  using ZA11 = ZeroAdapter<M11>;
  using ZA10 = ZeroAdapter<M10>;
  using ZA01 = ZeroAdapter<M01>;
  using ZA00 = ZeroAdapter<M00>;

  using ZA21 = ZeroAdapter<M21>;
  using ZA12 = ZeroAdapter<M12>;
  using ZA22 = ZeroAdapter<M22>;
  using ZA23 = ZeroAdapter<M23>;
  using ZA20 = ZeroAdapter<M20>;
  using ZA02 = ZeroAdapter<M02>;
  using ZA03 = ZeroAdapter<M03>;

  using ZA13 = ZeroAdapter<M13>;
  using ZA31 = ZeroAdapter<M31>;
  using ZA33 = ZeroAdapter<M33>;
  using ZA30 = ZeroAdapter<M30>;
}


TEST(eigen3, ConstantAdapter_functions)
{
  ConstantAdapter<M34, double, 5> c534 {};
  ConstantAdapter<M30, double, 5> c530_4 {4};
  ConstantAdapter<M04, double, 5> c504_3 {3};
  ConstantAdapter<M00, double, 5> c500_34 {3, 4};

  ConstantAdapter<M33, double, 5> c533 {};
  ConstantAdapter<M30, double, 5> c530_3 {3};
  ConstantAdapter<M03, double, 5> c503_3 {3};
  ConstantAdapter<M00, double, 5> c500_33 {3, 3};

  ConstantAdapter<M31, double, 5> c531 {};
  ConstantAdapter<M30, double, 5> c530_1 {1};
  ConstantAdapter<M01, double, 5> c501_3 {3};
  ConstantAdapter<M00, double, 5> c500_31 {3, 1};

  EXPECT_EQ(get_dimensions_of<0>(c534), 3);
  EXPECT_EQ(get_dimensions_of<0>(c530_4), 3);
  EXPECT_EQ(get_dimensions_of<0>(c504_3), 3);
  EXPECT_EQ(get_dimensions_of<0>(c500_34), 3);

  EXPECT_EQ(get_dimensions_of<1>(c534), 4);
  EXPECT_EQ(get_dimensions_of<1>(c530_4), 4);
  EXPECT_EQ(get_dimensions_of<1>(c504_3), 4);
  EXPECT_EQ(get_dimensions_of<1>(c500_34), 4);

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  EXPECT_TRUE(is_near(transpose(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c530_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c504_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c500_34), M43::Constant(5)));
  static_assert(eigen_constant_expr<decltype(transpose(c500_34))>);

  EXPECT_TRUE(is_near(adjoint(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c530_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c504_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c500_34), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(ConstantAdapter<CM34, cdouble, 5> {}), CM43::Constant(cdouble(5,0))));
  static_assert(eigen_constant_expr<decltype(adjoint(c500_34))>);

  EXPECT_NEAR(determinant(c533), 0, 1e-6);
  EXPECT_NEAR(determinant(c530_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c530_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c500_33), 0, 1e-6);

  EXPECT_NEAR(trace(c533), 15, 1e-6);
  EXPECT_NEAR(trace(c530_3), 15, 1e-6);
  EXPECT_NEAR(trace(c530_3), 15, 1e-6);
  EXPECT_NEAR(trace(c500_33), 15, 1e-6);

  // \todo rank_update

  M23 m23_66 = make_eigen_matrix<double, 2, 3>(6, 14, 22, 6, 14, 22);
  M20 m20_3_66 {2,3}; m20_3_66 = m23_66;
  M03 m03_2_66 {2,3}; m03_2_66 = m23_66;
  M00 m00_23_66 {2,3}; m00_23_66 = m23_66;
  auto m23_12 = make_eigen_matrix<double, 2, 3>(1.5, 3.5, 5.5, 1.5, 3.5, 5.5);

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, m00_23_66), m23_12));

  ConstantAdapter<M23, double, 8> c23_8;
  ConstantAdapter<M20, double, 8> c20_3_8 {3};
  ConstantAdapter<M03, double, 8> c03_2_8 {2};
  ConstantAdapter<M00, double, 8> c00_23_8 {2, 3};
  ConstantAdapter<M23, double, 2> c23_2;

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c00_23_8), c23_2));

  ConstantAdapter<M23, double, 6> c23_6;
  ConstantAdapter<M20, double, 6> c20_3_6 {3};
  ConstantAdapter<M03, double, 6> c03_2_6 {2};
  ConstantAdapter<M00, double, 6> c00_23_6 {2, 3};
  auto m23_15 = make_eigen_matrix<double, 2, 3>(1.5, 1.5, 1.5, 1.5, 1.5, 1.5);

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, double, 2> {}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, double, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, double, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {2, 2}, c00_23_6), m23_15));

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, double, 2> {}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, double, 2> {}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, double, 2> {}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, double, 2> {}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, double, 2> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, double, 2> {1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, double, 2> {1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, double, 2> {1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, double, 2> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, double, 2> {1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, double, 2> {1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, double, 2> {1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {1, 1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {1, 1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {1, 1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, double, 2> {1, 1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));

  EXPECT_TRUE(is_near(solve(M11::Identity(), make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(8)));

  EXPECT_TRUE(is_near(LQ_decomposition(ConstantAdapter<eigen_matrix_t<double, 5, 3>, double, 7> ()), LQ_decomposition(make_eigen_matrix<double, 5, 3>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto lq332 = make_self_contained(LQ_decomposition(make_eigen_matrix<double, 3, 2>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(LQ_decomposition(ConstantAdapter<M32, double, 3> ()), lq332));
  auto lqzc30_2 = LQ_decomposition(ConstantAdapter<M30, double, 3> {2});
  EXPECT_TRUE(is_near(lqzc30_2, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc30_2), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc30_2), 3);
  auto lqzc02_3 = LQ_decomposition(ConstantAdapter<M02, double, 3> {3});
  EXPECT_TRUE(is_near(lqzc02_3, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc02_3), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc02_3), 3);
  auto lqzc00_32 = LQ_decomposition(ConstantAdapter<M00, double, 3> {3, 2});
  EXPECT_TRUE(is_near(lqzc00_32, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc00_32), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc00_32), 3);

  EXPECT_TRUE(is_near(QR_decomposition(ConstantAdapter<eigen_matrix_t<double, 3, 5>, double, 7> ()), QR_decomposition(make_eigen_matrix<double, 3, 5>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto qr323 = make_self_contained(QR_decomposition(make_eigen_matrix<double, 2, 3>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(QR_decomposition(ConstantAdapter<M23, double, 3> ()), qr323));
  auto qrzc20_3 = QR_decomposition(ConstantAdapter<M20, double, 3> {3});
  EXPECT_TRUE(is_near(qrzc20_3, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc20_3), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc20_3), 3);
  auto qrzc03_2 = QR_decomposition(ConstantAdapter<M03, double, 3> {2});
  EXPECT_TRUE(is_near(qrzc03_2, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc03_2), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc03_2), 3);
  auto qrzc00_23 = QR_decomposition(ConstantAdapter<M00, double, 3> {2, 3});
  EXPECT_TRUE(is_near(qrzc00_23, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc00_23), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc00_23), 3);
}


TEST(eigen3, ZeroAdapter_functions)
{
  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z20_3 {Dimensions<2>(), 3};
  ZA03 z03_2 {2, Dimensions<3>()};
  ZA00 z00_23 {2, 3};

  ZA33 z33 {Dimensions<3>(), Dimensions<3>()};
  ZA30 z30_3 {Dimensions<3>(), 3};
  ZA03 z03_3 {3, Dimensions<3>()};
  ZA00 z00_33 {3, 3};

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_22 {2, 2};

  ZA21 z21;
  ZA20 z20_1 {1};
  ZA01 z01_2 {2};
  ZA00 z00_21 {2, 1};

  ZA12 z12;
  ZA10 z10_2 {2};
  ZA02 z02_1 {1};
  ZA00 z00_12 {1, 2};

  // transpose

  auto ez11 {M11::Identity() - M11::Identity()};

  auto ez21 {(M22::Identity() - M22::Identity()).diagonal()};
  auto ez01_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 1> {ez11, 2, 1};
  auto ez20_1 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 1};
  auto ez00_21 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 1};

  auto ez12 = Eigen::Replicate<decltype(ez11), 1, 2> {ez11, 1, 2};

  EXPECT_TRUE(is_near(transpose(ez21), ez12)); static_assert(zero_matrix<decltype(transpose(ez21))>);
  EXPECT_TRUE(is_near(transpose(ez20_1), ez12)); static_assert(zero_matrix<decltype(transpose(ez20_1))>);
  EXPECT_TRUE(is_near(transpose(ez01_2), ez12)); static_assert(zero_matrix<decltype(transpose(ez01_2))>);
  EXPECT_TRUE(is_near(transpose(ez00_21), ez12)); static_assert(zero_matrix<decltype(transpose(ez00_21))>);

  auto ec11_2 {M11::Identity() + M11::Identity()};

  auto ec21_2 = ec11_2.replicate<2, 1>();
  auto ec20_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 1);
  auto ec01_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1>(ec11_2, 2, 1);
  auto ec00_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 1);

  auto ec12_2 = Eigen::Replicate<decltype(ec11_2), 1, 2> {ec11_2, 1, 2};

  EXPECT_TRUE(is_near(transpose(ec21_2), ec12_2)); static_assert(constant_coefficient_v<decltype(transpose(ec21_2))> == 2);
  EXPECT_TRUE(is_near(transpose(ec20_1_2), ec12_2)); static_assert(constant_coefficient_v<decltype(transpose(ec20_1_2))> == 2);
  EXPECT_TRUE(is_near(transpose(ec01_2_2), ec12_2)); static_assert(constant_coefficient_v<decltype(transpose(ec01_2_2))> == 2);
  EXPECT_TRUE(is_near(transpose(ec00_21_2), ec12_2)); static_assert(constant_coefficient_v<decltype(transpose(ec00_21_2))> == 2);

  EXPECT_TRUE(is_near(transpose(z23), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z20_3), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z03_2), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z00_23), M32::Zero()));
  static_assert(zero_matrix<decltype(transpose(z00_23))>);

  // adjoint

  EXPECT_TRUE(is_near(adjoint(ec21_2), ec12_2)); static_assert(constant_coefficient_v<decltype(adjoint(ec21_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(ec20_1_2), ec12_2)); static_assert(constant_coefficient_v<decltype(adjoint(ec20_1_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(ec01_2_2), ec12_2)); static_assert(constant_coefficient_v<decltype(adjoint(ec01_2_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(ec00_21_2), ec12_2)); static_assert(constant_coefficient_v<decltype(adjoint(ec00_21_2))> == 2);

  EXPECT_TRUE(is_near(adjoint(ez21), ez12)); static_assert(zero_matrix<decltype(adjoint(ez21))>);
  EXPECT_TRUE(is_near(adjoint(ez20_1), ez12)); static_assert(zero_matrix<decltype(adjoint(ez20_1))>);
  EXPECT_TRUE(is_near(adjoint(ez01_2), ez12)); static_assert(zero_matrix<decltype(adjoint(ez01_2))>);
  EXPECT_TRUE(is_near(adjoint(ez00_21), ez12)); static_assert(zero_matrix<decltype(adjoint(ez00_21))>);

  EXPECT_TRUE(is_near(adjoint(z23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z20_3), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z03_2), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z00_23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(ZeroAdapter<CM23> {}), M32::Zero()));
  static_assert(zero_matrix<decltype(adjoint(z00_23))>);

  // determinant

  auto ez22 {M22::Identity() - M22::Identity()};
  auto ez02_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 2> {ez11, 2, 2};
  auto ez20_2 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 2};
  auto ez00_22 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 2};

  auto ec22_2 = ec11_2.replicate<2, 2>();
  auto ec20_2_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 2);
  auto ec02_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 2>(ec11_2, 2, 2);
  auto ec00_22_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 2);

  EXPECT_NEAR(determinant(ez22), 0, 1e-6);
  EXPECT_NEAR(determinant(ez20_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ez02_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ez00_22), 0, 1e-6);

  EXPECT_NEAR(determinant(z22), 0, 1e-6);
  EXPECT_NEAR(determinant(z20_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z02_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z00_22), 0, 1e-6);

  EXPECT_NEAR(determinant(ec22_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ec20_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ec02_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ec00_22_2), 0, 1e-6);
  EXPECT_NEAR(determinant(M22::Identity()), 1, 1e-6);

  // trace

  EXPECT_NEAR(trace(ez22), 0, 1e-6);
  EXPECT_NEAR(trace(ez20_2), 0, 1e-6);
  EXPECT_NEAR(trace(ez02_2), 0, 1e-6);
  EXPECT_NEAR(trace(ez00_22), 0, 1e-6);

  EXPECT_NEAR(trace(z22), 0, 1e-6);
  EXPECT_NEAR(trace(z20_2), 0, 1e-6);
  EXPECT_NEAR(trace(z02_2), 0, 1e-6);
  EXPECT_NEAR(trace(z00_22), 0, 1e-6);

  EXPECT_NEAR(trace(ec22_2), 4, 1e-6);
  EXPECT_NEAR(trace(ec20_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(ec02_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(ec00_22_2), 4, 1e-6);
  EXPECT_NEAR(trace(M22::Identity()), 2, 1e-6);

  // contract

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);

  EXPECT_TRUE(is_near(contract(ez22, m23), z23));
  EXPECT_TRUE(is_near(contract(ez20_2, m23), z23));
  EXPECT_TRUE(is_near(contract(ez02_2, m23), z23));
  EXPECT_TRUE(is_near(contract(ez00_22, m23), z23));
  EXPECT_TRUE(is_near(contract(m23, z33), z23));
  EXPECT_TRUE(is_near(contract(m23, z30_3), z23));
  EXPECT_TRUE(is_near(contract(m23, z03_3), z23));
  EXPECT_TRUE(is_near(contract(m23, z00_33), z23));
  static_assert(zero_matrix<decltype(contract(ez22, m23))>);
  static_assert(zero_matrix<decltype(contract(ez20_2, m23))>);
  static_assert(zero_matrix<decltype(contract(ez02_2, m23))>);
  static_assert(zero_matrix<decltype(contract(ez00_22, m23))>);
  static_assert(zero_matrix<decltype(contract(m23, z33))>);
  static_assert(zero_matrix<decltype(contract(m23, z30_3))>);
  static_assert(zero_matrix<decltype(contract(m23, z03_3))>);
  static_assert(zero_matrix<decltype(contract(m23, z00_33))>);

  EXPECT_TRUE(is_near(contract(m23, make_zero_matrix_like<M33>()), z23));
  EXPECT_TRUE(is_near(contract(make_zero_matrix_like<M22>(), m23), z23));
  static_assert(zero_matrix<decltype(contract(m23, make_zero_matrix_like<M33>()))>);
  static_assert(zero_matrix<decltype(contract(make_zero_matrix_like<M22>(), m23))>);

  auto ec23_2 = ec11_2.replicate<2, 3>();
  auto ec20_3_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 3);
  auto ec03_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 3>(ec11_2, 2, 3);
  auto ec00_23_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 3);

  auto ec11_3 {M11::Identity() + M11::Identity() + M11::Identity()};

  auto ec33_3 = ec11_3.replicate<3, 3>();
  auto ec30_3_3 = Eigen::Replicate<decltype(ec11_3), 3, Eigen::Dynamic>(ec11_3, 3, 3);
  auto ec03_3_3 = Eigen::Replicate<decltype(ec11_3), Eigen::Dynamic, 3>(ec11_3, 3, 3);
  auto ec00_33_3 = Eigen::Replicate<decltype(ec11_3), Eigen::Dynamic, Eigen::Dynamic>(ec11_3, 3, 3);

  auto ec23_18 = make_constant_matrix_like<M23, double, 18>();

  EXPECT_TRUE(is_near(contract(ec23_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec20_3_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec03_2_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec00_23_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec23_2, ec30_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec20_3_2, ec30_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec03_2_2, ec30_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec00_23_2, ec30_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec23_2, ec03_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec20_3_2, ec03_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec03_2_2, ec03_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec00_23_2, ec03_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec23_2, ec00_33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec20_3_2, ec00_33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec03_2_2, ec00_33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec00_23_2, ec00_33_3), ec23_18));
  static_assert(constant_coefficient_v<decltype(contract(ec23_2, ec33_3))> == 18);

  EXPECT_TRUE(is_near(contract(make_constant_matrix_like<M23, double, 2>(), make_constant_matrix_like<M33, double, 3>()), ec23_18));
  static_assert(constant_coefficient_v<decltype(contract(make_constant_matrix_like<M23, double, 2>(), make_constant_matrix_like<M33, double, 3>()))> == 18);

  EXPECT_TRUE(is_near(contract(m23, make_identity_matrix_like<M33>()), m23));
  EXPECT_TRUE(is_near(contract(make_identity_matrix_like<M22>(), m23), m23));
}


TEST(eigen3, constant_rank_update)
{
  auto c11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = Eigen::Replicate<decltype(c11_2), 2, 1> {c11_2, 2, 1}.asDiagonal();
  auto d20_1_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();
  auto d01_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1> {c11_2, 2, 1}.asDiagonal();
  auto d00_21_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();

  auto c11_3 {M11::Identity() + M11::Identity() + M11::Identity()};

  auto d21_3 = Eigen::Replicate<decltype(c11_3), 2, 1> {c11_3, 2, 1}.asDiagonal();
  auto d20_1_3 = Eigen::Replicate<decltype(c11_3), 2, Eigen::Dynamic> {c11_3, 2, 1}.asDiagonal();
  auto d01_2_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, 1> {c11_3, 2, 1}.asDiagonal();
  auto d00_21_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, Eigen::Dynamic> {c11_3, 2, 1}.asDiagonal();

  auto m22_5005 = make_dense_writable_matrix_from<M22>(5, 0, 0, 5);

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d20_1_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d01_2_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d00_21_3, d21_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d20_1_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d01_2_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d00_21_3, d20_1_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d20_1_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d01_2_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d00_21_3, d01_2_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d20_1_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d01_2_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d00_21_3, d00_21_2, 4), m22_5005));

  auto c11_9 {c11_3 + c11_3 + c11_3};

  auto d21_9 = Eigen::Replicate<decltype(c11_9), 2, 1> {c11_9, 2, 1}.asDiagonal();
  auto d20_1_9 = Eigen::Replicate<decltype(c11_9), 2, Eigen::Dynamic> {c11_9, 2, 1}.asDiagonal();
  auto d01_2_9 = Eigen::Replicate<decltype(c11_9), Eigen::Dynamic, 1> {c11_9, 2, 1}.asDiagonal();
  auto d00_21_9 = Eigen::Replicate<decltype(c11_9), Eigen::Dynamic, Eigen::Dynamic> {c11_9, 2, 1}.asDiagonal();

  auto m22_25 = make_dense_writable_matrix_from<M22>(25, 0, 0, 25);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(d21_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d20_1_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d01_2_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d00_21_9, d21_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(d21_9, d20_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d20_1_9, d20_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d01_2_9, d20_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d00_21_9, d20_1_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(d21_9, d01_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d20_1_9, d01_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d01_2_9, d01_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d00_21_9, d01_2_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(d21_9, d00_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d20_1_9, d00_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d01_2_9, d00_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d00_21_9, d00_21_2, 4), m22_25));

  auto m1034 = make_eigen_matrix<double, 2, 2>(1, 0, 3, 4);
  auto m1034_2 = m1034 * adjoint(m1034);

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_22 {2, 2};

  EXPECT_TRUE(is_near(rank_update_triangular(z22, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z20_2, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z02_2, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z00_22, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z22, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z20_2, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z02_2, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z00_22, m1034, 0.25), 0.25*m1034_2));

  auto di5 = M22::Identity() * 5;
  auto di5_2 = di5 * di5;

  EXPECT_TRUE(is_near(rank_update_triangular(z22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z20_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z02_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z00_22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z22, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z20_2, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z02_2, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z00_22, di5, 0.25), 0.25*di5_2));
}


TEST(eigen3, constant_solve)
{
  // B is zero

  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

  auto z11 = M11::Identity() - M11::Identity();

  auto z12 = Eigen::Replicate<decltype(z11), 1, 2> {z11, 1, 2};
  auto z10_2 = Eigen::Replicate<decltype(z11), 1, Eigen::Dynamic> {z11, 1, 2};
  auto z02_1 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 1, 2};
  auto z00_12 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 1, 2};

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(solve<true>(m22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(m00_22, z00_22), M22::Zero()));
  try { solve<true>(M12 {z12}, z12); EXPECT_TRUE(false); } catch (const std::runtime_error&) {}
  try { solve<true>(M00 {z12}, z00_12); EXPECT_TRUE(false); } catch (const std::runtime_error&) {}

  auto cd22 = M22::Identity() + M22::Identity();
  auto cd00_22 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, Eigen::Dynamic> {cd22, 1, 1};

  EXPECT_TRUE(is_near(solve<true>(cd22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(cd00_22, z00_22), M22::Zero()));

  auto c11 = M11::Identity() + M11::Identity();
  auto c12 = Eigen::Replicate<decltype(c11), 1, 2> {c11, 1, 2};
  auto c00_12 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 1, 2};

  EXPECT_TRUE(is_near(solve<true>(c12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(c00_12, z00_12), M22::Zero()));

  EXPECT_TRUE(is_near(solve(m22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z00_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z00_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z00_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z00_22), M22::Zero()));

  EXPECT_TRUE(is_near(solve(z12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z00_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z00_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z00_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z00_12), M22::Zero()));

  // A is zero

  auto m23_56 = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);
  auto m20_3_56 = M20 {m23_56};
  auto m03_2_56 = M03 {m23_56};
  auto m00_23_56 = M00 {m23_56};

  EXPECT_TRUE(is_near(solve(z22, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m00_23_56), M23::Zero()));

  // A and B are both constant

  auto c11_2 = M11::Identity() + M11::Identity();

  auto c22 = Eigen::Replicate<decltype(c11_2), 2, 2> {c11_2, 2, 2};
  auto c20_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 2};
  auto c02_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 2, 2};
  auto c00_22 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 2};

  auto c11_8 = c11_2 + c11_2 + c11_2 + c11_2;

  auto c23 = Eigen::Replicate<decltype(c11_8), 2, 3> {c11_8, 2, 3};
  auto c20_3 = Eigen::Replicate<decltype(c11_8), 2, Eigen::Dynamic> {c11_8, 2, 3};
  auto c03_2 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 3> {c11_8, 2, 3};
  auto c00_23 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 2, 3};

  auto m23_2 = make_dense_writable_matrix_from<M23>(2, 2, 2, 2, 2, 2);

  EXPECT_TRUE(is_near(solve(c22, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c00_23), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c00_23), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c00_23), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c00_23), m23_2));

  static_assert(constant_matrix<decltype(solve(c22, c23))>);
  static_assert(not constant_matrix<decltype(solve(c20_2, c23))>);
  static_assert(constant_matrix<decltype(solve(c02_2, c23))>);
  static_assert(not constant_matrix<decltype(solve(c00_22, c23))>);

  auto c12_2 = Eigen::Replicate<decltype(c11_2), 1, 2> {c11_2, 1, 2};
  auto c10_2_2 = Eigen::Replicate<decltype(c11_2), 1, Eigen::Dynamic> {c11_2, 1, 2};
  auto c02_1_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 1, 2};
  auto c00_12_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 1, 2};

  auto c12_8 = Eigen::Replicate<decltype(c11_8), 1, 2> {c11_8, 1, 2};
  auto c10_2_8 = Eigen::Replicate<decltype(c11_8), 1, Eigen::Dynamic> {c11_8, 1, 2};
  auto c02_1_8 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 2> {c11_8, 1, 2};
  auto c00_12_8 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 1, 2};

  auto m22_2 = make_dense_writable_matrix_from<M22>(2, 2, 2, 2);

  EXPECT_TRUE(is_near(solve(c12_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c00_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c00_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c00_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c00_12_8), m22_2));

  ZA23 z23;
  ZA20 z20_3 {3};
  ZA03 z03_2 {2};
  ZA00 z00_23 {2, 3};

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
}


TEST(eigen3, constant_decompositions)
{
  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZeroAdapter<M32> z32;
  ZeroAdapter<M30> z30_2 {2};
  ZeroAdapter<M02> z02_3 {3};
  ZA00 z00_32 {3, 2};

  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z20_3 {Dimensions<2>(), 3};
  ZA03 z03_2 {2, Dimensions<3>()};
  ZA00 z00_23 {2, 3};

  EXPECT_TRUE(is_near(LQ_decomposition(z23), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z20_3), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z03_2), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z00_23), z22));
  static_assert(zero_matrix<decltype(LQ_decomposition(z00_23))>);

  EXPECT_TRUE(is_near(QR_decomposition(z32), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z30_2), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z02_3), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z00_32), z22));
  static_assert(zero_matrix<decltype(QR_decomposition(z00_32))>);
}


TEST(eigen3, constant_diagonalizing)
{
  auto ez11 = M11::Identity() - M11::Identity();

  auto ez22 = M22::Identity() - M22::Identity();
  auto ez20_2 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 2};
  auto ez02_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 2> {ez11, 2, 2};
  auto ez00_22 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 2};

  auto ez21 = (M22::Identity() - M22::Identity()).diagonal();
  auto ez01_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 1> {ez11, 2, 1};
  auto ez20_1 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 1};
  auto ez00_21 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 1};

  EXPECT_TRUE(is_near(diagonal_of(ez22), ez21));
  EXPECT_TRUE(is_near(diagonal_of(ez20_2), ez21)); static_assert(zero_matrix<decltype(diagonal_of(ez20_2))>); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(ez20_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ez02_2), ez21)); static_assert(zero_matrix<decltype(diagonal_of(ez02_2))>); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(ez02_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ez00_22), ez21)); static_assert(zero_matrix<decltype(diagonal_of(ez00_22))>); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ez00_22))>);

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ez21}), ez21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper {ez21}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ez20_1}), ez21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper {ez20_1}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ez01_2}), ez21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper {ez01_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ez00_21}), ez21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper {ez00_21}))>);

  auto ec11_2 {M11::Identity() + M11::Identity()};

  auto ec21_2 = ec11_2.replicate<2, 1>();
  auto ec20_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 1);
  auto ec01_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1>(ec11_2, 2, 1);
  auto ec00_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 1);

  EXPECT_TRUE(is_near(diagonal_of(ec21_2.asDiagonal()), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec21_2.asDiagonal()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(ec20_1_2.asDiagonal()), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec20_1_2.asDiagonal()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(ec01_2_2.asDiagonal()), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec01_2_2.asDiagonal()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(ec00_21_2.asDiagonal()), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec00_21_2.asDiagonal()))> == 2);

  auto ec22_2 = ec11_2.replicate<2, 2>();
  auto ec20_2_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 2);
  auto ec02_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 2>(ec11_2, 2, 2);
  auto ec00_22_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 2);

  EXPECT_TRUE(is_near(diagonal_of(ec22_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec22_2))> == 2); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(ec22_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ec20_2_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec20_2_2))> == 2); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(ec20_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ec02_2_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec02_2_2))> == 2); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(ec02_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ec00_22_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(ec00_22_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ec00_22_2))>);

  auto d21_2 = ec21_2.asDiagonal();
  auto d20_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();
  auto d01_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1> {ec11_2, 2, 1}.asDiagonal();
  auto d00_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();

  EXPECT_TRUE(is_near(diagonal_of(d21_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d21_2))> == 2); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(d21_2))>);
  EXPECT_TRUE(is_near(diagonal_of(d20_1_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d20_1_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(d20_1_2))>);
  EXPECT_TRUE(is_near(diagonal_of(d01_2_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d01_2_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(d01_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(d00_21_2), ec21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d00_21_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(d00_21_2))>);

  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(diagonal_of(z22.template selfadjointView<Eigen::Upper>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z22.template selfadjointView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z20_2.template selfadjointView<Eigen::Lower>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z20_2.template selfadjointView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z02_2.template selfadjointView<Eigen::Upper>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z02_2.template selfadjointView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z00_22.template selfadjointView<Eigen::Lower>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z00_22.template selfadjointView<Eigen::Lower>()))>);

  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c22_2 = c11_2.replicate<2, 2>();
  auto c20_2_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 2);
  auto c02_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2>(c11_2, 2, 2);
  auto c00_22_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 2);

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template selfadjointView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c22_2.template selfadjointView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2.template selfadjointView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c20_2_2.template selfadjointView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2.template selfadjointView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c02_2_2.template selfadjointView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2.template selfadjointView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c00_22_2.template selfadjointView<Eigen::Lower>()))> == 2);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template selfadjointView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity().template selfadjointView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()))> == 1);

  ConstantAdapter<M33, double, 5> c533 {};
  ConstantAdapter<M30, double, 5> c530_3 {3};
  ConstantAdapter<M03, double, 5> c503_3 {3};
  ConstantAdapter<M00, double, 5> c500_33 {3, 3};

  ConstantAdapter<M31, double, 5> c531 {};
  ConstantAdapter<M30, double, 5> c530_1 {1};
  ConstantAdapter<M01, double, 5> c501_3 {3};
  ConstantAdapter<M00, double, 5> c500_31 {3, 1};

  auto m33_d5 = make_dense_writable_matrix_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5);

  EXPECT_TRUE(is_near(to_diagonal(c531), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c530_1), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c501_3), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c500_31), m33_d5));

  EXPECT_TRUE(is_near(diagonal_of(c533), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c530_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c503_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c500_33), M31::Constant(5)));
  static_assert(constant_coefficient_v<decltype(diagonal_of(c500_33))> == 5);

  ZA21 z21;
  ZA20 z20_1 {1};
  ZA01 z01_2 {2};
  ZA00 z00_21 {2, 1};

  ZA12 z12;
  ZA10 z10_2 {2};
  ZA02 z02_1 {1};
  ZA00 z00_12 {1, 2};

  EXPECT_TRUE(is_near(to_diagonal(z21), z22));
  EXPECT_TRUE(is_near(to_diagonal(z20_1), z22));
  EXPECT_TRUE(is_near(to_diagonal(z01_2), z22));
  EXPECT_TRUE(is_near(to_diagonal(z00_21), z22));
  static_assert(diagonal_adapter<decltype(to_diagonal(z21))>);
  static_assert(diagonal_adapter<decltype(to_diagonal(z01_2))>);
  static_assert(diagonal_adapter<decltype(to_diagonal(z20_1)), Likelihood::maybe>);
  static_assert(diagonal_matrix<decltype(to_diagonal(z20_1))>);
  static_assert(diagonal_adapter<decltype(to_diagonal(z00_21)), Likelihood::maybe>);
  static_assert(diagonal_matrix<decltype(to_diagonal(z00_21))>);
  static_assert(zero_matrix<decltype(to_diagonal(z00_21))>);

  EXPECT_TRUE(is_near(diagonal_of(z22), z21));
  EXPECT_TRUE(is_near(diagonal_of(z20_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(z02_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(z00_22), z21));
  static_assert(zero_matrix<decltype(diagonal_of(z00_22))>);
}


TEST(eigen3, constant_elementwise)
{
  auto m23 = make_dense_writable_matrix_from<M23>(5.5, 5.5, 5.5, 5.5, 5.5, 5.5);
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions{2}, Dimensions<3>{}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions<2>{}, Dimensions{3}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions{2}, Dimensions{3}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23>([]{return 5.5;}), m23));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));
}


TEST(eigen3, constant_reductions)
{
  // reduce zero

  auto ez11 = M11::Identity() - M11::Identity();
  using EZ11 = decltype(ez11);

  auto ez23 = Eigen::Replicate<EZ11, 2, 3> {ez11};
  auto ez20_3 = Eigen::Replicate<EZ11, 2, Eigen::Dynamic> {ez11, 2, 3};
  auto ez03_2 = Eigen::Replicate<EZ11, Eigen::Dynamic, 3> {ez11, 2, 3};
  auto ez00_23 = Eigen::Replicate<EZ11, Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 3};

  auto ez13 = Eigen::Replicate<EZ11, 1, 3> {ez11};
  auto ez21 = Eigen::Replicate<EZ11, 2, 1> {ez11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ez23), ez13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, ez23))>);
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ez03_2), ez13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, ez03_2))>);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ez20_3), ez13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, ez20_3))>);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ez00_23), ez13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, ez00_23))>);

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ez23), ez21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, ez23))>);
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ez03_2), ez21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, ez03_2))>);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ez20_3), ez21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, ez20_3))>);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ez00_23), ez21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, ez00_23))>);

  EXPECT_EQ((reduce(std::plus<double>{}, ez23)), 0);
  EXPECT_EQ((reduce(std::plus<double>{}, ez03_2)), 0);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, ez23), ez11));
  EXPECT_EQ((reduce(std::multiplies<double>{}, ez00_23)), 0);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, ez20_3), ez11));

  static_assert(reduce(std::plus<double>{}, ez11) == 0);
  static_assert(reduce(std::plus<double>{}, ez13) == 0);
  static_assert(reduce(std::plus<double>{}, ez21) == 0);
  static_assert(reduce(std::plus<double>{}, ez23) == 0);
  static_assert(reduce(std::multiplies<double>{}, ez23) == 0);

  // reduce constant

  auto ec11 = M11::Identity() + M11::Identity();
  using EC11 = decltype(ec11);

  auto ec23 = Eigen::Replicate<EC11, 2, 3> {ec11};
  auto ec20_3 = Eigen::Replicate<EC11, 2, Eigen::Dynamic> {ec11, 2, 3};
  auto ec03_2 = Eigen::Replicate<EC11, Eigen::Dynamic, 3> {ec11, 2, 3};
  auto ec00_23 = Eigen::Replicate<EC11, Eigen::Dynamic, Eigen::Dynamic> {ec11, 2, 3};

  auto ec13 = Eigen::Replicate<EC11, 1, 3> {ec11};
  auto ec21 = Eigen::Replicate<EC11, 2, 1> {ec11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ec23), 2 * ec13));
  static_assert(constant_coefficient_v<decltype(reduce<0>(std::plus<double>{}, ec23))> == 4);
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ec03_2), 2 * ec13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ec20_3), 2 * ec13));
  static_assert(constant_coefficient_v<decltype(reduce<0>(std::plus<double>{}, ec20_3))> == 4);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ec00_23), 2 * ec13));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ec23), 3 * ec21));
  static_assert(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, ec23))> == 6);
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ec03_2), 3 * ec21));
  static_assert(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, ec03_2))> == 6);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ec20_3), 4 * ec21));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ec00_23), 4 * ec21));

  EXPECT_EQ((reduce(std::plus<double>{}, ec23)), 12);
  EXPECT_EQ((reduce(std::plus<double>{}, ec03_2)), 12);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, ec23), 6 * ec11));
  EXPECT_EQ((reduce(std::multiplies<double>{}, ec00_23)), 64);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, ec20_3), 32 * ec11));

  double non_constexpr_dummy = 0.0;
  EXPECT_TRUE(is_near(reduce<1, 0>([&](auto a, auto b){ return non_constexpr_dummy + a + b; }, ec23), 6 * ec11));

  static_assert(reduce(std::plus<double>{}, ec11) == 2);
  static_assert(reduce(std::multiplies<double>{}, ec11) == 2);
  static_assert(reduce(std::plus<double>{}, ec13) == 6);
  static_assert(reduce(std::multiplies<double>{}, ec13) == 8);
  static_assert(reduce(std::plus<double>{}, ec21) == 4);
  static_assert(reduce(std::multiplies<double>{}, ec21) == 4);
  static_assert(reduce(std::plus<double>{}, ec23) == 12);
  static_assert(reduce(std::multiplies<double>{}, ec23) == 64);

  EXPECT_NEAR(reduce([&](auto a, auto b){ return non_constexpr_dummy + a * b; }, ec13), 8, 1e-9);

  // reduce constant (fractional)

  auto efc11 = (M11::Identity() + M11::Identity()).array() / (M11::Identity() + M11::Identity() + M11::Identity()).array();
  using EFC11 = decltype(efc11);
  static_assert(are_within_tolerance(constant_coefficient_v<EFC11>, 2./3));

  auto efc23 = Eigen::Replicate<EFC11, 2, 3> {efc11};
  auto efc20_3 = Eigen::Replicate<EFC11, 2, Eigen::Dynamic> {efc11, 2, 3};
  auto efc03_2 = Eigen::Replicate<EFC11, Eigen::Dynamic, 3> {efc11, 2, 3};
  auto efc00_23 = Eigen::Replicate<EFC11, Eigen::Dynamic, Eigen::Dynamic> {efc11, 2, 3};

  auto efc13 = Eigen::Replicate<EFC11, 1, 3> {efc11};
  auto efc21 = Eigen::Replicate<EFC11, 2, 1> {efc11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, efc23), 2 * efc13));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, efc03_2), 2 * efc13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, efc20_3), 2./3 * efc13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, efc00_23), 2./3 * efc13));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, efc23), 3 * efc21)); static_assert(are_within_tolerance(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, efc23))>, 2));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, efc03_2), 3 * efc21)); static_assert(are_within_tolerance(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, efc03_2))>, 2));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, efc20_3), 4./9 * efc21));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, efc00_23), 4./9 * efc21));

  EXPECT_EQ((reduce(std::plus<double>{}, efc23)), 4);
  EXPECT_EQ((reduce(std::plus<double>{}, efc03_2)), 4);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, efc23), 6 * efc11));
  EXPECT_NEAR((reduce(std::multiplies<double>{}, efc00_23)), 64./729, 1e-9);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, efc20_3), 32./243 * efc11));

  static_assert(are_within_tolerance(reduce(std::plus<double>{}, efc11), 2./3));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, efc11), 2./3));
  static_assert(are_within_tolerance(reduce(std::plus<double>{}, efc13), 2));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, efc13), 8./27));
  static_assert(are_within_tolerance(reduce(std::plus<double>{}, efc21), 4./3));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, efc21), 4./9));
  static_assert(are_within_tolerance(reduce(std::plus<double>{}, efc23), 4));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, efc23), 64./729));

  // average_reduce zero

  auto ez22 = M22::Identity() - M22::Identity();
  auto ez20_2 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 2};
  auto ez02_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 2> {ez11, 2, 2};
  auto ez00_22 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 2};

  EXPECT_TRUE(is_near(average_reduce<1>(ez22), ez21)); static_assert(zero_matrix<decltype(average_reduce<1>(ez22))>);
  EXPECT_TRUE(is_near(average_reduce<1>(ez20_2), ez21)); static_assert(zero_matrix<decltype(average_reduce<1>(ez20_2))>);
  EXPECT_TRUE(is_near(average_reduce<1>(ez02_2), ez21)); static_assert(zero_matrix<decltype(average_reduce<1>(ez02_2))>);
  EXPECT_TRUE(is_near(average_reduce<1>(ez00_22), ez21)); static_assert(zero_matrix<decltype(average_reduce<1>(ez00_22))>);

  auto ez12 = Eigen::Replicate<decltype(ez11), 1, 2> {ez11, 1, 2};

  EXPECT_TRUE(is_near(average_reduce<0>(ez22), ez12)); static_assert(zero_matrix<decltype(average_reduce<0>(ez22))>);
  EXPECT_TRUE(is_near(average_reduce<0>(ez20_2), ez12)); static_assert(zero_matrix<decltype(average_reduce<0>(ez20_2))>);
  EXPECT_TRUE(is_near(average_reduce<0>(ez02_2), ez12)); static_assert(zero_matrix<decltype(average_reduce<0>(ez02_2))>);
  EXPECT_TRUE(is_near(average_reduce<0>(ez00_22), ez12)); static_assert(zero_matrix<decltype(average_reduce<0>(ez00_22))>);

  static_assert(average_reduce(ez22) == 0);
  static_assert(average_reduce(ez20_2) == 0);
  static_assert(average_reduce(ez02_2) == 0);
  static_assert(average_reduce(ez00_22) == 0);

  // average_reduce constant

  EXPECT_TRUE(is_near(average_reduce<1>(ec23), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(ec23))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(ec20_3), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(ec20_3))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(ec03_2), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(ec03_2))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(ec00_23), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(ec00_23))> == 2);

  EXPECT_TRUE(is_near(average_reduce<0>(ec23), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(ec23))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(ec20_3), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(ec20_3))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(ec03_2), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(ec03_2))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(ec00_23), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(ec00_23))> == 2);

  static_assert(average_reduce(ec23) == 2);
  static_assert(average_reduce(ec20_3) == 2);
  static_assert(average_reduce(ec03_2) == 2);
  static_assert(average_reduce(ec00_23) == 2);

  // average_reduce constant diagonal

  auto ec11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = ec11_2.replicate<2, 1>().asDiagonal();
  auto d20_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();
  auto d01_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1> {ec11_2, 2, 1}.asDiagonal();
  auto d00_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();

  static_assert(constant_coefficient_v<decltype(average_reduce<1>(d21_2))> == 1);
  EXPECT_TRUE(is_near(average_reduce<1>(d20_1_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<1>(d01_2_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<1>(d00_21_2), M21::Constant(1)));

  static_assert(constant_coefficient_v<decltype(average_reduce<0>(d21_2))> == 1);
  EXPECT_TRUE(is_near(average_reduce<0>(d20_1_2), M12::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<0>(d01_2_2), M12::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<0>(d00_21_2), M12::Constant(1)));

  static_assert(constant_coefficient_v<decltype(average_reduce<0>(d21_2))> == 1);
  static_assert(constant_coefficient_v<decltype(average_reduce<1>(d21_2))> == 1);

  static_assert(average_reduce(d21_2) == 1);
  EXPECT_EQ(average_reduce(d20_1_2), 1);
  EXPECT_EQ(average_reduce(d01_2_2), 1);
  EXPECT_EQ(average_reduce(d00_21_2), 1);

  // average_reduce identity

  auto i21 = M22::Identity();
  auto i20_1 = Eigen::Replicate<typename M11::IdentityReturnType, 2, Eigen::Dynamic> {M11::Identity(), 2, 1}.asDiagonal();
  auto i01_2 = Eigen::Replicate<typename M11::IdentityReturnType, Eigen::Dynamic, 1> {M11::Identity(), 2, 1}.asDiagonal();
  auto i00_21 = Eigen::Replicate<typename M11::IdentityReturnType, Eigen::Dynamic, Eigen::Dynamic> {M11::Identity(), 2, 1}.asDiagonal();
  static_assert(identity_matrix<decltype(i21)>);
  static_assert(identity_matrix<decltype(i20_1)>);
  static_assert(identity_matrix<decltype(i01_2)>);
  static_assert(identity_matrix<decltype(i00_21)>);

  EXPECT_TRUE(is_near(average_reduce<1>(i21), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i20_1), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i01_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i00_21), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(average_reduce<0>(i21), M12::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(i20_1), M12::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(i01_2), M12::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(i00_21), M12::Constant(0.5)));

  static_assert(average_reduce(i21) == 0.5);
  EXPECT_EQ(average_reduce(i20_1), 0.5);
  EXPECT_EQ(average_reduce(i01_2), 0.5);
  EXPECT_EQ(average_reduce(i00_21), 0.5);

  // reduce ZeroAdapter

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_22 {2, 2};

  EXPECT_TRUE(is_near(average_reduce<1>(z22), ez21));
  EXPECT_TRUE(is_near(average_reduce<1>(z20_2), ez21));
  EXPECT_TRUE(is_near(average_reduce<1>(z02_2), ez21));
  EXPECT_TRUE(is_near(average_reduce<1>(z00_22), ez21));
  static_assert(zero_matrix<decltype(average_reduce<1>(z00_22))>);

  EXPECT_TRUE(is_near(average_reduce<0>(z22), ez12));
  EXPECT_TRUE(is_near(average_reduce<0>(z20_2), ez12));
  EXPECT_TRUE(is_near(average_reduce<0>(z02_2), ez12));
  EXPECT_TRUE(is_near(average_reduce<0>(z00_22), ez12));
  static_assert(zero_matrix<decltype(average_reduce<0>(z00_22))>);

  // reduce ConstantAdapter

  ConstantAdapter<M34, double, 5> c534 {};
  ConstantAdapter<M30, double, 5> c530_4 {4};
  ConstantAdapter<M04, double, 5> c504_3 {3};
  ConstantAdapter<M00, double, 5> c500_34 {3, 4};

  ConstantAdapter<M33, double, 5> c533 {};
  ConstantAdapter<M30, double, 5> c530_3 {3};
  ConstantAdapter<M03, double, 5> c503_3 {3};
  ConstantAdapter<M00, double, 5> c500_33 {3, 3};

  auto colzc34 = average_reduce<1>(c500_34);
  EXPECT_TRUE(is_near(average_reduce<1>(ConstantAdapter<M23, double, 3> ()), (M21::Constant(3))));
  EXPECT_EQ(colzc34, (ConstantAdapter<M31, double, 5> {}));
  EXPECT_EQ(get_dimensions_of<0>(colzc34), 3);
  EXPECT_EQ(get_dimensions_of<1>(colzc34), 1);
  static_assert(eigen_constant_expr<decltype(colzc34)>);

  auto rowzc34 = average_reduce<0>(c500_34);
  EXPECT_TRUE(is_near(average_reduce<0>(ConstantAdapter<M23, double, 3> ()), (M13::Constant(3))));
  EXPECT_EQ(rowzc34, (ConstantAdapter<eigen_matrix_t<double, 1, 4>, double, 5> {}));
  EXPECT_EQ(get_dimensions_of<1>(rowzc34), 4);
  EXPECT_EQ(get_dimensions_of<0>(rowzc34), 1);
  static_assert(eigen_constant_expr<decltype(rowzc34)>);
}


TEST(eigen3, constant_element_functions)
{
  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_22 {2, 2};

  ZA21 z21;
  ZA20 z20_1 {1};
  ZA01 z01_2 {2};
  ZA00 z00_21 {2, 1};

  ZA12 z12;
  ZA10 z10_2 {2};
  ZA02 z02_1 {1};
  ZA00 z00_12 {1, 2};

  // get_element
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

  EXPECT_NEAR(get_element(ConstantAdapter<M22, double, 5> {}, 1, 0), 5, 1e-8);

  ConstantAdapter<M00, double, 5> c00 {2, 2};

  EXPECT_NEAR((get_element(c00, 0, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 0, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(ConstantAdapter<M01, double, 7> {3}, 0)), 7, 1e-6);

  // get_block

  std::integral_constant<std::size_t, 0> N0;
  std::integral_constant<std::size_t, 1> N1;
  std::integral_constant<std::size_t, 2> N2;
  std::integral_constant<std::size_t, 3> N3;

  auto z34 = make_zero_matrix_like<M34>();

  EXPECT_TRUE(is_near(get_block(z34, std::tuple{N0, N0}, std::tuple{N2, N2}), make_zero_matrix_like<M22>()));
  EXPECT_TRUE(is_near(get_block(z34, std::tuple{N1, N1}, std::tuple{2, N3}), make_zero_matrix_like<M23>()));
  EXPECT_TRUE(is_near(get_block(z34, std::tuple{0, 0}, std::tuple{2, N2}), make_zero_matrix_like<M22>()));
  EXPECT_TRUE(is_near(get_block(z34, std::tuple{0, 0}, std::tuple{2, N2}), make_zero_matrix_like<M22>()));
  EXPECT_TRUE(is_near(get_block<1, 0>(z34, std::tuple{N0, N0}, std::tuple{N3, N2}), make_zero_matrix_like<M23>()));
  EXPECT_TRUE(is_near(get_block<0>(z34, std::tuple{N0}, std::tuple{N2}), make_zero_matrix_like<M24>()));
  EXPECT_TRUE(is_near(get_block<0>(z34, std::tuple{1}, std::tuple{1}), make_zero_matrix_like<M14>()));
  EXPECT_TRUE(is_near(get_block<1>(z34, std::tuple{N0}, std::tuple{N2}), make_zero_matrix_like<M32>()));
  EXPECT_TRUE(is_near(get_block<1>(z34, std::tuple{1}, std::tuple{1}), make_zero_matrix_like<M31>()));

  auto c34 = make_constant_matrix_like<M34, 3>();
  EXPECT_TRUE(is_near(get_block(c34, std::tuple{N0, N0}, std::tuple{N2, N2}), make_constant_matrix_like<M22, 3>()));
  EXPECT_TRUE(is_near(get_block(c34, std::tuple{N1, N1}, std::tuple{2, N3}), make_constant_matrix_like<M23, 3>()));
  EXPECT_TRUE(is_near(get_block(c34, std::tuple{0, 0}, std::tuple{2, N2}), make_constant_matrix_like<M22, 3>()));
  EXPECT_TRUE(is_near(get_block(c34, std::tuple{0, 0}, std::tuple{2, N2}), make_constant_matrix_like<M22, 3>()));
  EXPECT_TRUE(is_near(get_block<1, 0>(c34, std::tuple{N0, N0}, std::tuple{N3, N2}), make_constant_matrix_like<M23, 3>()));
  EXPECT_TRUE(is_near(get_block<0>(c34, std::tuple{N0}, std::tuple{N2}), make_constant_matrix_like<M24, 3>()));
  EXPECT_TRUE(is_near(get_block<0>(c34, std::tuple{1}, std::tuple{1}), make_constant_matrix_like<M14, 3>()));
  EXPECT_TRUE(is_near(get_block<1>(c34, std::tuple{N0}, std::tuple{N2}), make_constant_matrix_like<M32, 3>()));
  EXPECT_TRUE(is_near(get_block<1>(c34, std::tuple{1}, std::tuple{1}), make_constant_matrix_like<M31, 3>()));

  // get_chip

  ConstantAdapter<M00, double, 5> c500_34 {3, 4};

  EXPECT_TRUE(is_near(get_chip<1>(ConstantAdapter<M23, double, 6> {}, 1), (M21::Constant(6))));
  EXPECT_TRUE(is_near(get_chip<1>(ConstantAdapter<M23, double, 7> {}, N1), (M21::Constant(7))));
  auto c5c34 = get_chip<1>(c500_34, 1);
  EXPECT_EQ(get_dimensions_of<0>(c5c34), 3);
  static_assert(get_dimensions_of<1>(c5c34) == 1);
  static_assert(constant_coefficient_v<decltype(c5c34)> == 5);
  auto c5v34 = get_chip<1>(ConstantAdapter<M04, double, 5> {3}, N1);
  EXPECT_EQ(get_dimensions_of<0>(c5v34), 3);
  static_assert(get_dimensions_of<1>(c5v34) == 1);
  static_assert(constant_coefficient_v<decltype(c5v34)> == 5);

  EXPECT_TRUE(is_near(get_chip<0>(ConstantAdapter<M32, double, 6> {}, 1), (M12::Constant(6))));
  EXPECT_TRUE(is_near(get_chip<0>(ConstantAdapter<M32, double, 7> {}, N1), (M12::Constant(7))));
  auto r5c34 = get_chip<0>(c500_34, 1);
  EXPECT_EQ(get_dimensions_of<1>(r5c34), 4);
  static_assert(get_dimensions_of<0>(r5c34) == 1);
  static_assert(constant_coefficient_v<decltype(r5c34)> == 5);
  auto r5v34 = get_chip<0>(ConstantAdapter<M30, double, 5> {4}, N1);
  EXPECT_EQ(get_dimensions_of<1>(r5v34), 4);
  static_assert(get_dimensions_of<0>(r5v34) == 1);
  static_assert(constant_coefficient_v<decltype(r5v34)> == 5);

  auto zc34 = ZeroAdapter<M34> {};

  EXPECT_TRUE(is_near(get_column(ZeroAdapter<M23>(), N1), (M21::Zero())));
  EXPECT_TRUE(is_near(get_column(ZeroAdapter<M23>(), 1), (M21::Zero())));
  auto czc34 = get_column(zc34, 1);
  EXPECT_EQ(get_dimensions_of<0>(czc34), 3);
  static_assert(get_dimensions_of<1>(czc34) == 1);
  static_assert(zero_matrix<decltype(czc34)>);
  auto czv34 = get_column(ZeroAdapter<M04> {3}, N1);
  EXPECT_EQ(get_dimensions_of<0>(czv34), 3);
  static_assert(get_dimensions_of<1>(czv34) == 1);
  static_assert(zero_matrix<decltype(czv34)>);

  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), N1), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M30>(Dimensions<3>{}, 3), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M03>(3, Dimensions<3>{}), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M00>(3,3), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));

  EXPECT_TRUE(is_near(get_row(ZeroAdapter<M32>(Dimensions<3>{}, Dimensions<2>{}), N1), (M12::Zero())));
  EXPECT_TRUE(is_near(get_row(ZeroAdapter<M32>(Dimensions<3>{}, Dimensions<2>{}), 1), (M12::Zero())));
  auto rzc34 = get_row(zc34, 1);
  EXPECT_EQ(get_dimensions_of<1>(rzc34), 4);
  static_assert(get_dimensions_of<0>(rzc34) == 1);
  static_assert(zero_matrix<decltype(rzc34)>);
  auto rzv34 = get_row(ZeroAdapter<M30> {Dimensions<3>{}, 4}, N1);
  EXPECT_EQ(get_dimensions_of<1>(rzv34), 4);
  static_assert(get_dimensions_of<0>(rzv34) == 1);
  static_assert(zero_matrix<decltype(rzv34)>);

  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), N1), make_dense_writable_matrix_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M30>(Dimensions<3>{}, 3), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M03>(3, Dimensions<3>{}), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M00>(3,3), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));

  // \todo tile

  // \todo concatenate


  auto tup_z33_z23 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M23>()};

  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(Dimensions<5>{}, Dimensions<3>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(Dimensions<5>{}, 3), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(5, Dimensions<3>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(5, 3), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));

  auto tup_z33_z32 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M32>()};

  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(Dimensions<3>{}, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(Dimensions<3>{}, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(3, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(3, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));

  auto tup_z33_z22 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M22>()};

  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M55>(Dimensions<5>{}, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M50>(Dimensions<5>{}, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M05>(5, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M00>(5, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));

  auto tup_z32_z23 = std::tuple {make_zero_matrix_like<M32>(), make_zero_matrix_like<M23>()};

  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M55>(Dimensions<5>{}, Dimensions<5>{}), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M50>(Dimensions<5>{}, 5), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M05>(5, Dimensions<5>{}), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M00>(5, 5), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
}


TEST(eigen3, constant_chipwise_operations)
{
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + make_constant_matrix_like<double, 1>(row)); }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + make_constant_matrix_like<double, 1>(row)); }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));

  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + make_constant_matrix_like<double, 1>(col)); }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + make_constant_matrix_like<double, 1>(col)); }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));
}

