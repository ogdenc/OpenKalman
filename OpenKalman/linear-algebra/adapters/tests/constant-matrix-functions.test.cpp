/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "adapters.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using stdex::numbers::pi;

namespace
{
  using Axis2 = Dimensions<2>;

  using ZA11 = zero_adapter<M11>;
  using ZA10 = zero_adapter<M1x>;
  using ZA01 = zero_adapter<Mx1>;
  using ZA00 = zero_adapter<Mxx>;

  using ZA21 = zero_adapter<M21>;
  using ZA12 = zero_adapter<M12>;
  using ZA22 = zero_adapter<M22>;
  using ZA23 = zero_adapter<M23>;
  using ZA20 = zero_adapter<M2x>;
  using ZA02 = zero_adapter<Mx2>;
  using ZA03 = zero_adapter<Mx3>;

  using ZA13 = zero_adapter<M13>;
  using ZA31 = zero_adapter<M31>;
  using ZA33 = zero_adapter<M33>;
  using ZA30 = zero_adapter<M3x>;
}


TEST(adapters, constant_adapter_functions)
{
  constant_adapter<M34, double, 5> c534 {};
  constant_adapter<M3x, double, 5> c53x_4 {4};
  constant_adapter<Mx4, double, 5> c5x4_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_34 {3, 4};

  constant_adapter<M33, double, 5> c533 {};
  constant_adapter<M3x, double, 5> c53x_3 {3};
  constant_adapter<Mx3, double, 5> c5x3_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_33 {3, 3};

  constant_adapter<M31, double, 5> c531 {};
  constant_adapter<M3x, double, 5> c53x_1 {1};
  constant_adapter<Mx1, double, 5> c5x1_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_31 {3, 1};

  EXPECT_EQ(get_pattern_collection<0>(c534), 3);
  EXPECT_EQ(get_pattern_collection<0>(c53x_4), 3);
  EXPECT_EQ(get_pattern_collection<0>(c5x4_3), 3);
  EXPECT_EQ(get_pattern_collection<0>(c5xx_34), 3);

  EXPECT_EQ(get_pattern_collection<1>(c534), 4);
  EXPECT_EQ(get_pattern_collection<1>(c53x_4), 4);
  EXPECT_EQ(get_pattern_collection<1>(c5x4_3), 4);
  EXPECT_EQ(get_pattern_collection<1>(c5xx_34), 4);

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  EXPECT_TRUE(is_near(transpose(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c53x_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c5x4_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c5xx_34), M43::Constant(5)));
  static_assert(constant_matrix<decltype(transpose(c5xx_34))>);

  EXPECT_TRUE(is_near(adjoint(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c53x_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c5x4_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c5xx_34), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(constant_adapter<CM34, cdouble, 5> {}), CM43::Constant(cdouble(5,0))));
  static_assert(constant_matrix<decltype(adjoint(c5xx_34))>);

  EXPECT_NEAR(determinant(c533), 0, 1e-6);
  EXPECT_NEAR(determinant(c53x_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c53x_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c5xx_33), 0, 1e-6);

  EXPECT_NEAR(trace(c533), 15, 1e-6);
  EXPECT_NEAR(trace(c53x_3), 15, 1e-6);
  EXPECT_NEAR(trace(c53x_3), 15, 1e-6);
  EXPECT_NEAR(trace(c5xx_33), 15, 1e-6);

  // \todo rank_update

  M23 m23_66 = make_dense_object_from<M23>(6, 14, 22, 6, 14, 22);
  M2x m20_3_66 {2,3}; m20_3_66 = m23_66;
  Mx3 m03_2_66 {2,3}; m03_2_66 = m23_66;
  Mxx m00_23_66 {2,3}; m00_23_66 = m23_66;
  auto m23_12 = make_dense_object_from<M23>(1.5, 3.5, 5.5, 1.5, 3.5, 5.5);

  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, m00_23_66), m23_12));

  constant_adapter<M23, double, 8> c23_8;
  constant_adapter<M2x, double, 8> c2x_3_8 {3};
  constant_adapter<Mx3, double, 8> cx3_2_8 {2};
  constant_adapter<Mxx, double, 8> cxx_23_8 {2, 3};
  constant_adapter<M23, double, 2> c23_2;

  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, cxx_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, c2x_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, cx3_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, cxx_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, c2x_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, cx3_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, c2x_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, cx3_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, cxx_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, c2x_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, cx3_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, cxx_23_8), c23_2));

  constant_adapter<M23, double, 6> c23_6;
  constant_adapter<M2x, double, 6> c2x_3_6 {3};
  constant_adapter<Mx3, double, 6> cx3_2_6 {2};
  constant_adapter<Mxx, double, 6> cxx_23_6 {2, 3};
  auto m23_15 = make_dense_object_from<M23>(1.5, 1.5, 1.5, 1.5, 1.5, 1.5);

  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, cxx_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, c2x_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M22, double, 2> {}, cx3_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, cxx_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, c2x_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<M2x, double, 2> {2}, cx3_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, c2x_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, cx3_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx2, double, 2> {2}, cxx_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, c2x_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, cx3_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {2, 2}, cxx_23_6), m23_15));

  auto m11_8 = M11 {8};
  EXPECT_TRUE(is_near(solve(constant_adapter<M11, double, 2> {}, m11_8), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M11, double, 2> {}, M1x(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M11, double, 2> {}, Mx1(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M11, double, 2> {}, Mxx(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M1x, double, 2> {1}, m11_8), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M1x, double, 2> {1}, M1x(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M1x, double, 2> {1}, Mx1(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<M1x, double, 2> {1}, Mxx(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx1, double, 2> {1}, m11_8), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx1, double, 2> {1}, M1x(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx1, double, 2> {1}, Mx1(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mx1, double, 2> {1}, Mxx(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {1, 1}, m11_8), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {1, 1}, M1x(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {1, 1}, Mx1(m11_8)), M11(4)));
  EXPECT_TRUE(is_near(solve(constant_adapter<Mxx, double, 2> {1, 1}, Mxx(m11_8)), M11(4)));

  EXPECT_TRUE(is_near(solve(M11::Identity(), make_dense_object_from<M11>(8)), make_dense_object_from<M11>(8)));

  EXPECT_TRUE(is_near(LQ_decomposition(constant_adapter<eigen_matrix_t<double, 5, 3>, double, 7> ()), to_dense_object(LQ_decomposition(M53::Constant(7))).cwiseAbs()));
  auto lq332 = make_self_contained(to_dense_object(LQ_decomposition(M32::Constant(3))).cwiseAbs());
  EXPECT_TRUE(is_near(LQ_decomposition(constant_adapter<M32, double, 3> ()), lq332));
  auto lqzc3x_2 = LQ_decomposition(constant_adapter<M3x, double, 3> {2});
  EXPECT_TRUE(is_near(lqzc3x_2, lq332));
  EXPECT_EQ(get_pattern_collection<0>(lqzc3x_2), 3);
  EXPECT_EQ(get_pattern_collection<1>(lqzc3x_2), 3);
  auto lqzcx2_3 = LQ_decomposition(constant_adapter<Mx2, double, 3> {3});
  EXPECT_TRUE(is_near(lqzcx2_3, lq332));
  EXPECT_EQ(get_pattern_collection<0>(lqzcx2_3), 3);
  EXPECT_EQ(get_pattern_collection<1>(lqzcx2_3), 3);
  auto lqzcxx_32 = LQ_decomposition(constant_adapter<Mxx, double, 3> {3, 2});
  EXPECT_TRUE(is_near(lqzcxx_32, lq332));
  EXPECT_EQ(get_pattern_collection<0>(lqzcxx_32), 3);
  EXPECT_EQ(get_pattern_collection<1>(lqzcxx_32), 3);

  EXPECT_TRUE(is_near(LQ_decomposition(constant_adapter<M32, double> (3.)), lq332));
  EXPECT_TRUE(is_near(LQ_decomposition(constant_adapter<M3x, double> (3., 2)), lq332));
  EXPECT_TRUE(is_near(LQ_decomposition(constant_adapter<Mx2, double> (3., 3)), lq332));
  EXPECT_TRUE(is_near(LQ_decomposition(constant_adapter<Mxx, double> (3., 3, 2)), lq332));

  EXPECT_TRUE(is_near(QR_decomposition(constant_adapter<eigen_matrix_t<double, 3, 5>, double, 7> ()), to_dense_object(QR_decomposition(M35::Constant(7))).cwiseAbs()));
  auto qr323 = make_self_contained(to_dense_object(QR_decomposition(M23::Constant(3))).cwiseAbs());
  EXPECT_TRUE(is_near(QR_decomposition(constant_adapter<M23, double, 3> ()), qr323));
  auto qrzc2x_3 = QR_decomposition(constant_adapter<M2x, double, 3> {3});
  EXPECT_TRUE(is_near(qrzc2x_3, qr323));
  EXPECT_EQ(get_pattern_collection<0>(qrzc2x_3), 3);
  EXPECT_EQ(get_pattern_collection<1>(qrzc2x_3), 3);
  auto qrzcx3_2 = QR_decomposition(constant_adapter<Mx3, double, 3> {2});
  EXPECT_TRUE(is_near(qrzcx3_2, qr323));
  EXPECT_EQ(get_pattern_collection<0>(qrzcx3_2), 3);
  EXPECT_EQ(get_pattern_collection<1>(qrzcx3_2), 3);
  auto qrzcxx_23 = QR_decomposition(constant_adapter<Mxx, double, 3> {2, 3});
  EXPECT_TRUE(is_near(qrzcxx_23, qr323));
  EXPECT_EQ(get_pattern_collection<0>(qrzcxx_23), 3);
  EXPECT_EQ(get_pattern_collection<1>(qrzcxx_23), 3);

  EXPECT_TRUE(is_near(QR_decomposition(constant_adapter<M23, double> (3.)), qr323));
  EXPECT_TRUE(is_near(QR_decomposition(constant_adapter<M2x, double> (3., 3)), qr323));
  EXPECT_TRUE(is_near(QR_decomposition(constant_adapter<Mx3, double> (3., 2)), qr323));
  EXPECT_TRUE(is_near(QR_decomposition(constant_adapter<Mxx, double> (3., 2, 3)), qr323));
}


TEST(adapters, zero_adapter_functions)
{
  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z2x_3 {Dimensions<2>(), 3};
  ZA03 zx3_2 {2, Dimensions<3>()};
  ZA00 zxx_23 {2, 3};

  ZA33 z33 {Dimensions<3>(), Dimensions<3>()};
  ZA30 z3x_3 {Dimensions<3>(), 3};
  ZA03 zx3_3 {3, Dimensions<3>()};
  ZA00 zxx_33 {3, 3};

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z2x_2 {Dimensions<2>(), 2};
  ZA02 zx2_2 {2, Dimensions<2>()};
  ZA00 zxx_22 {2, 2};

  ZA21 z21;
  ZA20 z2x_1 {1};
  ZA01 zx1_2 {2};
  ZA00 zxx_21 {2, 1};

  ZA12 z12;
  ZA10 z1x_2 {2};
  ZA02 zx2_1 {1};
  ZA00 zxx_12 {1, 2};

  // transpose

  auto ez11 {M11::Identity() - M11::Identity()};

  auto ez21 {(M22::Identity() - M22::Identity()).diagonal()};
  auto ezx1_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 1> {ez11, 2, 1};
  auto ez2x_1 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 1};
  auto ezxx_21 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 1};

  auto ez12 = Eigen::Replicate<decltype(ez11), 1, 2> {ez11, 1, 2};

  EXPECT_TRUE(is_near(transpose(ez21), ez12)); static_assert(zero<decltype(transpose(ez21))>);
  EXPECT_TRUE(is_near(transpose(ez2x_1), ez12)); static_assert(zero<decltype(transpose(ez2x_1))>);
  EXPECT_TRUE(is_near(transpose(ezx1_2), ez12)); static_assert(zero<decltype(transpose(ezx1_2))>);
  EXPECT_TRUE(is_near(transpose(ezxx_21), ez12)); static_assert(zero<decltype(transpose(ezxx_21))>);

  auto ec11_2 {M11::Identity() + M11::Identity()};

  auto ec21_2 = ec11_2.replicate<2, 1>();
  auto ec2x_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 1);
  auto ecx1_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1>(ec11_2, 2, 1);
  auto ecxx_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 1);

  auto ec12_2 = Eigen::Replicate<decltype(ec11_2), 1, 2> {ec11_2, 1, 2};

  EXPECT_TRUE(is_near(transpose(ec21_2), ec12_2)); static_assert(constant_value_v<decltype(transpose(ec21_2))> == 2);
  EXPECT_TRUE(is_near(transpose(ec2x_1_2), ec12_2)); static_assert(constant_value_v<decltype(transpose(ec2x_1_2))> == 2);
  EXPECT_TRUE(is_near(transpose(ecx1_2_2), ec12_2)); static_assert(constant_value_v<decltype(transpose(ecx1_2_2))> == 2);
  EXPECT_TRUE(is_near(transpose(ecxx_21_2), ec12_2)); static_assert(constant_value_v<decltype(transpose(ecxx_21_2))> == 2);

  EXPECT_TRUE(is_near(transpose(z23), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z2x_3), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(zx3_2), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(zxx_23), M32::Zero()));
  static_assert(zero<decltype(transpose(zxx_23))>);

  // adjoint

  EXPECT_TRUE(is_near(adjoint(ec21_2), ec12_2)); static_assert(constant_value_v<decltype(adjoint(ec21_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(ec2x_1_2), ec12_2)); static_assert(constant_value_v<decltype(adjoint(ec2x_1_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(ecx1_2_2), ec12_2)); static_assert(constant_value_v<decltype(adjoint(ecx1_2_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(ecxx_21_2), ec12_2)); static_assert(constant_value_v<decltype(adjoint(ecxx_21_2))> == 2);

  EXPECT_TRUE(is_near(adjoint(ez21), ez12)); static_assert(zero<decltype(adjoint(ez21))>);
  EXPECT_TRUE(is_near(adjoint(ez2x_1), ez12)); static_assert(zero<decltype(adjoint(ez2x_1))>);
  EXPECT_TRUE(is_near(adjoint(ezx1_2), ez12)); static_assert(zero<decltype(adjoint(ezx1_2))>);
  EXPECT_TRUE(is_near(adjoint(ezxx_21), ez12)); static_assert(zero<decltype(adjoint(ezxx_21))>);

  EXPECT_TRUE(is_near(adjoint(z23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z2x_3), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(zx3_2), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(zxx_23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(zero_adapter<CM23> {}), M32::Zero()));
  static_assert(zero<decltype(adjoint(zxx_23))>);

  // determinant

  auto ez22 {M22::Identity() - M22::Identity()};
  auto ezx2_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 2> {ez11, 2, 2};
  auto ez2x_2 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 2};
  auto ezxx_22 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 2};

  auto ec22_2 = ec11_2.replicate<2, 2>();
  auto ec2x_2_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 2);
  auto ecx2_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 2>(ec11_2, 2, 2);
  auto ecxx_22_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 2);

  EXPECT_NEAR(determinant(ez22), 0, 1e-6);
  EXPECT_NEAR(determinant(ez2x_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ezx2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ezxx_22), 0, 1e-6);

  EXPECT_NEAR(determinant(z22), 0, 1e-6);
  EXPECT_NEAR(determinant(z2x_2), 0, 1e-6);
  EXPECT_NEAR(determinant(zx2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(zxx_22), 0, 1e-6);

  EXPECT_NEAR(determinant(ec22_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ec2x_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ecx2_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(ecxx_22_2), 0, 1e-6);
  EXPECT_NEAR(determinant(M22::Identity()), 1, 1e-6);

  // trace

  EXPECT_NEAR(trace(ez22), 0, 1e-6);
  EXPECT_NEAR(trace(ez2x_2), 0, 1e-6);
  EXPECT_NEAR(trace(ezx2_2), 0, 1e-6);
  EXPECT_NEAR(trace(ezxx_22), 0, 1e-6);

  EXPECT_NEAR(trace(z22), 0, 1e-6);
  EXPECT_NEAR(trace(z2x_2), 0, 1e-6);
  EXPECT_NEAR(trace(zx2_2), 0, 1e-6);
  EXPECT_NEAR(trace(zxx_22), 0, 1e-6);

  EXPECT_NEAR(trace(ec22_2), 4, 1e-6);
  EXPECT_NEAR(trace(ec2x_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(ecx2_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(ecxx_22_2), 4, 1e-6);
  EXPECT_NEAR(trace(M22::Identity()), 2, 1e-6);

  // contract

  auto m23 = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);

  EXPECT_TRUE(is_near(contract(ez22, m23), z23));
  EXPECT_TRUE(is_near(contract(ez2x_2, m23), z23));
  EXPECT_TRUE(is_near(contract(ezx2_2, m23), z23));
  EXPECT_TRUE(is_near(contract(ezxx_22, m23), z23));
  EXPECT_TRUE(is_near(contract(m23, z33), z23));
  EXPECT_TRUE(is_near(contract(m23, z3x_3), z23));
  EXPECT_TRUE(is_near(contract(m23, zx3_3), z23));
  EXPECT_TRUE(is_near(contract(m23, zxx_33), z23));
  static_assert(zero<decltype(contract(ez22, m23))>);
  static_assert(zero<decltype(contract(ez2x_2, m23))>);
  static_assert(zero<decltype(contract(ezx2_2, m23))>);
  static_assert(zero<decltype(contract(ezxx_22, m23))>);
  static_assert(zero<decltype(contract(m23, z33))>);
  static_assert(zero<decltype(contract(m23, z3x_3))>);
  static_assert(zero<decltype(contract(m23, zx3_3))>);
  static_assert(zero<decltype(contract(m23, zxx_33))>);

  EXPECT_TRUE(is_near(contract(m23, make_zero<M33>()), z23));
  EXPECT_TRUE(is_near(contract(make_zero<M22>(), m23), z23));
  static_assert(zero<decltype(contract(m23, make_zero<M33>()))>);
  static_assert(zero<decltype(contract(make_zero<M22>(), m23))>);

  auto ec23_2 = ec11_2.replicate<2, 3>();
  auto ec2x_3_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 3);
  auto ecx3_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 3>(ec11_2, 2, 3);
  auto ecxx_23_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 3);

  auto ec11_3 {M11::Identity() + M11::Identity() + M11::Identity()};

  auto ec33_3 = ec11_3.replicate<3, 3>();
  auto ec3x_3_3 = Eigen::Replicate<decltype(ec11_3), 3, Eigen::Dynamic>(ec11_3, 3, 3);
  auto ecx3_3_3 = Eigen::Replicate<decltype(ec11_3), Eigen::Dynamic, 3>(ec11_3, 3, 3);
  auto ecxx_33_3 = Eigen::Replicate<decltype(ec11_3), Eigen::Dynamic, Eigen::Dynamic>(ec11_3, 3, 3);

  auto ec23_18 = make_constant<M23, double, 18>();

  EXPECT_TRUE(is_near(contract(ec23_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec2x_3_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecx3_2_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecxx_23_2, ec33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec23_2, ec3x_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec2x_3_2, ec3x_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecx3_2_2, ec3x_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecxx_23_2, ec3x_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec23_2, ecx3_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec2x_3_2, ecx3_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecx3_2_2, ecx3_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecxx_23_2, ecx3_3_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec23_2, ecxx_33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ec2x_3_2, ecxx_33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecx3_2_2, ecxx_33_3), ec23_18));
  EXPECT_TRUE(is_near(contract(ecxx_23_2, ecxx_33_3), ec23_18));
  static_assert(constant_value_v<decltype(contract(ec23_2, ec33_3))> == 18);

  EXPECT_TRUE(is_near(contract(make_constant<M23, double, 2>(), make_constant<M33, double, 3>()), ec23_18));
  static_assert(constant_value_v<decltype(contract(make_constant<M23, double, 2>(), make_constant<M33, double, 3>()))> == 18);

  EXPECT_TRUE(is_near(contract(m23, make_identity_matrix_like(m33)), m23));
  EXPECT_TRUE(is_near(contract(make_identity_matrix_like(m22), m23), m23));
}


TEST(adapters, constant_rank_update)
{
  auto c11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = Eigen::Replicate<decltype(c11_2), 2, 1> {c11_2, 2, 1}.asDiagonal();
  auto d2x_1_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();
  auto dx1_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1> {c11_2, 2, 1}.asDiagonal();
  auto dxx_21_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();

  auto c11_3 {M11::Identity() + M11::Identity() + M11::Identity()};

  auto d21_3 = Eigen::Replicate<decltype(c11_3), 2, 1> {c11_3, 2, 1}.asDiagonal();
  auto d2x_1_3 = Eigen::Replicate<decltype(c11_3), 2, Eigen::Dynamic> {c11_3, 2, 1}.asDiagonal();
  auto dx1_2_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, 1> {c11_3, 2, 1}.asDiagonal();
  auto dxx_21_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, Eigen::Dynamic> {c11_3, 2, 1}.asDiagonal();

  auto m22_5005 = make_dense_object_from<M22>(5, 0, 0, 5);

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d2x_1_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dx1_2_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dxx_21_3, d21_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d2x_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d2x_1_3, d2x_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dx1_2_3, d2x_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dxx_21_3, d2x_1_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, dx1_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d2x_1_3, dx1_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dx1_2_3, dx1_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dxx_21_3, dx1_2_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, dxx_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d2x_1_3, dxx_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dx1_2_3, dxx_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(dxx_21_3, dxx_21_2, 4), m22_5005));

  auto c11_9 {c11_3 + c11_3 + c11_3};

  auto d21_9 = Eigen::Replicate<decltype(c11_9), 2, 1> {c11_9, 2, 1}.asDiagonal();
  auto d20_1_9 = Eigen::Replicate<decltype(c11_9), 2, Eigen::Dynamic> {c11_9, 2, 1}.asDiagonal();
  auto d01_2_9 = Eigen::Replicate<decltype(c11_9), Eigen::Dynamic, 1> {c11_9, 2, 1}.asDiagonal();
  auto d00_21_9 = Eigen::Replicate<decltype(c11_9), Eigen::Dynamic, Eigen::Dynamic> {c11_9, 2, 1}.asDiagonal();

  auto m22_25 = make_dense_object_from<M22>(25, 0, 0, 25);

  EXPECT_TRUE(is_near(rank_update_hermitian(d21_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d20_1_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d01_2_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d00_21_9, d21_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_hermitian(d21_9, d2x_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d20_1_9, d2x_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d01_2_9, d2x_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d00_21_9, d2x_1_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_hermitian(d21_9, dx1_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d20_1_9, dx1_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d01_2_9, dx1_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d00_21_9, dx1_2_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_hermitian(d21_9, dxx_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d20_1_9, dxx_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d01_2_9, dxx_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_hermitian(d00_21_9, dxx_21_2, 4), m22_25));

  auto m1034 = make_dense_object_from<M22>(1, 0, 3, 4);
  auto m1034_2 = m1034 * adjoint(m1034);

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z2x_2 {Dimensions<2>(), 2};
  ZA02 zx2_2 {2, Dimensions<2>()};
  ZA00 zxx_22 {2, 2};

  EXPECT_TRUE(is_near(rank_update_triangular(z22, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z2x_2, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(zx2_2, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(zxx_22, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_hermitian(z22, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_hermitian(z2x_2, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_hermitian(zx2_2, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_hermitian(zxx_22, m1034, 0.25), 0.25*m1034_2));

  auto di5 = M22::Identity() * 5;
  auto di5_2 = di5 * di5;

  EXPECT_TRUE(is_near(rank_update_triangular(z22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z2x_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(zx2_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(zxx_22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_hermitian(z22, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_hermitian(z2x_2, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_hermitian(zx2_2, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_hermitian(zxx_22, di5, 0.25), 0.25*di5_2));
}


TEST(adapters, constant_solve)
{
  // B is zero

  auto m22 = make_dense_object_from<M22>(1, 2, 3, 4);
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  auto z11 = M11::Identity() - M11::Identity();

  auto z12 = Eigen::Replicate<decltype(z11), 1, 2> {z11, 1, 2};
  auto z1x_2 = Eigen::Replicate<decltype(z11), 1, Eigen::Dynamic> {z11, 1, 2};
  auto zx2_1 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 1, 2};
  auto zxx_12 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 1, 2};

  auto z22 = M22::Identity() - M22::Identity();
  auto z2x_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto zx2_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto zxx_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(solve<true>(m22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(mxx_22, zxx_22), M22::Zero()));
  try { solve<true>(M12 {z12}, z12); EXPECT_TRUE(false); } catch (const std::runtime_error&) {}
  try { solve<true>(Mxx {z12}, zxx_12); EXPECT_TRUE(false); } catch (const std::runtime_error&) {}

  auto cd22 = M22::Identity() + M22::Identity();
  auto cd00_22 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, Eigen::Dynamic> {cd22, 1, 1};

  EXPECT_TRUE(is_near(solve<true>(cd22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(cd00_22, zxx_22), M22::Zero()));

  auto c11 = M11::Identity() + M11::Identity();
  auto c12 = Eigen::Replicate<decltype(c11), 1, 2> {c11, 1, 2};
  auto cxx_12 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 1, 2};

  EXPECT_TRUE(is_near(solve<true>(c12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(cxx_12, zxx_12), M22::Zero()));

  EXPECT_TRUE(is_near(solve(m22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m2x_2, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mx2_2, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mxx_22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z2x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m2x_2, z2x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mx2_2, z2x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mxx_22, z2x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, zx2_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m2x_2, zx2_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mx2_2, zx2_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mxx_22, zx2_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, zxx_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m2x_2, zxx_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mx2_2, zxx_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(mxx_22, zxx_22), M22::Zero()));

  EXPECT_TRUE(is_near(solve(z12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z1x_2, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_1, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z1x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z1x_2, z1x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_1, z1x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_12, z1x_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, zx2_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z1x_2, zx2_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_1, zx2_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_12, zx2_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, zxx_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z1x_2, zxx_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_1, zxx_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_12, zxx_12), M22::Zero()));

  // A is zero

  auto m23_56 = make_dense_object_from<M23>(5, 7, 9, 6, 8, 10);
  auto m20_3_56 = M2x {m23_56};
  auto m03_2_56 = Mx3 {m23_56};
  auto m00_23_56 = Mxx {m23_56};

  EXPECT_TRUE(is_near(solve(z22, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z2x_2, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z2x_2, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z2x_2, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z2x_2, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_2, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_2, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_2, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zx2_2, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_22, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_22, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_22, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(zxx_22, m00_23_56), M23::Zero()));

  // A and B are both constant

  auto c11_2 = M11::Identity() + M11::Identity();

  auto c22 = Eigen::Replicate<decltype(c11_2), 2, 2> {c11_2, 2, 2};
  auto c2x_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 2};
  auto cx2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 2, 2};
  auto cxx_22 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 2};

  auto c11_8 = c11_2 + c11_2 + c11_2 + c11_2;

  auto c23 = Eigen::Replicate<decltype(c11_8), 2, 3> {c11_8, 2, 3};
  auto c2x_3 = Eigen::Replicate<decltype(c11_8), 2, Eigen::Dynamic> {c11_8, 2, 3};
  auto cx3_2 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 3> {c11_8, 2, 3};
  auto cxx_23 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 2, 3};

  auto m23_2 = make_dense_object_from<M23>(2, 2, 2, 2, 2, 2);

  EXPECT_TRUE(is_near(solve(c22, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c2x_3), m23_2));
  EXPECT_TRUE(is_near(solve(c22, cx3_2), m23_2));
  EXPECT_TRUE(is_near(solve(c22, cxx_23), m23_2));
  EXPECT_TRUE(is_near(solve(c2x_2, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c2x_2, c2x_3), m23_2));
  EXPECT_TRUE(is_near(solve(c2x_2, cx3_2), m23_2));
  EXPECT_TRUE(is_near(solve(c2x_2, cxx_23), m23_2));
  EXPECT_TRUE(is_near(solve(cx2_2, c23), m23_2));
  EXPECT_TRUE(is_near(solve(cx2_2, c2x_3), m23_2));
  EXPECT_TRUE(is_near(solve(cx2_2, cx3_2), m23_2));
  EXPECT_TRUE(is_near(solve(cx2_2, cxx_23), m23_2));
  EXPECT_TRUE(is_near(solve(cxx_22, c23), m23_2));
  EXPECT_TRUE(is_near(solve(cxx_22, c2x_3), m23_2));
  EXPECT_TRUE(is_near(solve(cxx_22, cx3_2), m23_2));
  EXPECT_TRUE(is_near(solve(cxx_22, cxx_23), m23_2));

  static_assert(constant_value_v<decltype(solve(c22, c23))> == 2);
  static_assert(values::dynamic<constant_value<decltype(solve(c2x_2, c23))>>);
  static_assert(constant_value_v<decltype(solve(cx2_2, c23))> == 2);
  static_assert(values::dynamic<constant_value<decltype(solve(cxx_22, c23))>>);

  auto c12_2 = Eigen::Replicate<decltype(c11_2), 1, 2> {c11_2, 1, 2};
  auto c1x_2_2 = Eigen::Replicate<decltype(c11_2), 1, Eigen::Dynamic> {c11_2, 1, 2};
  auto cx2_1_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 1, 2};
  auto cxx_12_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 1, 2};

  auto c12_8 = Eigen::Replicate<decltype(c11_8), 1, 2> {c11_8, 1, 2};
  auto c1x_2_8 = Eigen::Replicate<decltype(c11_8), 1, Eigen::Dynamic> {c11_8, 1, 2};
  auto cx2_1_8 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 2> {c11_8, 1, 2};
  auto cxx_12_8 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 1, 2};

  auto m22_2 = make_dense_object_from<M22>(2, 2, 2, 2);

  EXPECT_TRUE(is_near(solve(c12_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c1x_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, cx2_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, cxx_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c1x_2_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c1x_2_2, c1x_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c1x_2_2, cx2_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c1x_2_2, cxx_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(cx2_1_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(cx2_1_2, c1x_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(cx2_1_2, cx2_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(cx2_1_2, cxx_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(cxx_12_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(cxx_12_2, c1x_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(cxx_12_2, cx2_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(cxx_12_2, cxx_12_8), m22_2));

  ZA23 z23;
  ZA20 z2x_3 {3};
  ZA03 zx3_2 {2};
  ZA00 zxx_23 {2, 3};

  EXPECT_TRUE(is_near(solve(z22, z23), z23));
  EXPECT_TRUE(is_near(solve(z22, zxx_23), z23));
  EXPECT_TRUE(is_near(solve(z22, z2x_3), z23));
  EXPECT_TRUE(is_near(solve(z22, zx3_2), z23));
  EXPECT_TRUE(is_near(solve(z2x_2, z23), z23));
  EXPECT_TRUE(is_near(solve(z2x_2, zxx_23), z23));
  EXPECT_TRUE(is_near(solve(z2x_2, z2x_3), z23));
  EXPECT_TRUE(is_near(solve(z2x_2, zx3_2), z23));
  EXPECT_TRUE(is_near(solve(zx2_2, z23), z23));
  EXPECT_TRUE(is_near(solve(zx2_2, z2x_3), z23));
  EXPECT_TRUE(is_near(solve(zx2_2, zx3_2), z23));
  EXPECT_TRUE(is_near(solve(zx2_2, zxx_23), z23));
  EXPECT_TRUE(is_near(solve(zxx_22, z23), z23));
  EXPECT_TRUE(is_near(solve(zxx_22, z2x_3), z23));
  EXPECT_TRUE(is_near(solve(zxx_22, zx3_2), z23));
  EXPECT_TRUE(is_near(solve(zxx_22, zxx_23), z23));
}


TEST(adapters, constant_decompositions)
{
  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  zero_adapter<M32> z32;
  zero_adapter<M3x> z3x_2 {2};
  zero_adapter<Mx2> zx2_3 {3};
  ZA00 zxx_32 {3, 2};

  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z2x_3 {Dimensions<2>(), 3};
  ZA03 zx3_2 {2, Dimensions<3>()};
  ZA00 zxx_23 {2, 3};

  EXPECT_TRUE(is_near(LQ_decomposition(z23), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z2x_3), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(zx3_2), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(zxx_23), z22));
  static_assert(zero<decltype(LQ_decomposition(zxx_23))>);

  EXPECT_TRUE(is_near(QR_decomposition(z32), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z3x_2), z22));
  EXPECT_TRUE(is_near(QR_decomposition(zx2_3), z22));
  EXPECT_TRUE(is_near(QR_decomposition(zxx_32), z22));
  static_assert(zero<decltype(QR_decomposition(zxx_32))>);
}


TEST(adapters, constant_diagonalizing)
{
  auto ez11 = M11::Identity() - M11::Identity();

  auto ez22 = M22::Identity() - M22::Identity();
  auto ez2x_2 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 2};
  auto ezx2_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 2> {ez11, 2, 2};
  auto ezxx_22 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 2};

  auto ez21 = (M22::Identity() - M22::Identity()).diagonal();
  auto ezx1_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 1> {ez11, 2, 1};
  auto ez2x_1 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 1};
  auto ezxx_21 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 1};

  EXPECT_TRUE(is_near(diagonal_of(ez22), ez21));
  EXPECT_TRUE(is_near(diagonal_of(ez2x_2), ez21)); static_assert(zero<decltype(diagonal_of(ez2x_2))>); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ez2x_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ezx2_2), ez21)); static_assert(zero<decltype(diagonal_of(ezx2_2))>); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ezx2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ezxx_22), ez21)); static_assert(zero<decltype(diagonal_of(ezxx_22))>); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ezxx_22))>);

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ez21}), ez21)); static_assert(zero<decltype(diagonal_of(Eigen::DiagonalWrapper {ez21}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ez2x_1}), ez21)); static_assert(zero<decltype(diagonal_of(Eigen::DiagonalWrapper {ez2x_1}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ezx1_2}), ez21)); static_assert(zero<decltype(diagonal_of(Eigen::DiagonalWrapper {ezx1_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {ezxx_21}), ez21)); static_assert(zero<decltype(diagonal_of(Eigen::DiagonalWrapper {ezxx_21}))>);

  auto ec11_2 {M11::Identity() + M11::Identity()};

  auto ec21_2 = ec11_2.replicate<2, 1>();
  auto ec2x_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 1);
  auto ecx1_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1>(ec11_2, 2, 1);
  auto ecxx_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 1);

  EXPECT_TRUE(is_near(diagonal_of(ec21_2.asDiagonal()), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ec21_2.asDiagonal()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(ec2x_1_2.asDiagonal()), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ec2x_1_2.asDiagonal()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(ecx1_2_2.asDiagonal()), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ecx1_2_2.asDiagonal()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(ecxx_21_2.asDiagonal()), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ecxx_21_2.asDiagonal()))> == 2);

  auto ec22_2 = ec11_2.replicate<2, 2>();
  auto ec2x_2_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic>(ec11_2, 2, 2);
  auto ecx2_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 2>(ec11_2, 2, 2);
  auto ecxx_22_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic>(ec11_2, 2, 2);

  EXPECT_TRUE(is_near(diagonal_of(ec22_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ec22_2))> == 2); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(ec22_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ec2x_2_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ec2x_2_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ec2x_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ecx2_2_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ecx2_2_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ecx2_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(ecxx_22_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(ecxx_22_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(ecxx_22_2))>);

  auto d21_2 = ec21_2.asDiagonal();
  auto d2x_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();
  auto dx1_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1> {ec11_2, 2, 1}.asDiagonal();
  auto dxx_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();

  EXPECT_TRUE(is_near(diagonal_of(d21_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(d21_2))> == 2); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(d21_2))>);
  EXPECT_TRUE(is_near(diagonal_of(d2x_1_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(d2x_1_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(d2x_1_2))>);
  EXPECT_TRUE(is_near(diagonal_of(dx1_2_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(dx1_2_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(dx1_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(dxx_21_2), ec21_2)); static_assert(constant_value_v<decltype(diagonal_of(dxx_21_2))> == 2); static_assert(has_dynamic_dimensions<decltype(diagonal_of(dxx_21_2))>);

  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z2x_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto zx2_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto zxx_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(diagonal_of(z22.template selfadjointView<Eigen::Upper>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(z22.template selfadjointView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z2x_2.template selfadjointView<Eigen::Lower>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(z2x_2.template selfadjointView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(zx2_2.template selfadjointView<Eigen::Upper>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(zx2_2.template selfadjointView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(zxx_22.template selfadjointView<Eigen::Lower>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(zxx_22.template selfadjointView<Eigen::Lower>()))>);

  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c22_2 = c11_2.replicate<2, 2>();
  auto c2x_2_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 2);
  auto cx2_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2>(c11_2, 2, 2);
  auto cxx_22_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 2);

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template selfadjointView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(c22_2.template selfadjointView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c2x_2_2.template selfadjointView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(c2x_2_2.template selfadjointView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(cx2_2_2.template selfadjointView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(cx2_2_2.template selfadjointView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(cxx_22_2.template selfadjointView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(cxx_22_2.template selfadjointView<Eigen::Lower>()))> == 2);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template selfadjointView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M22::Identity().template selfadjointView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template selfadjointView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2,2).template selfadjointView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template selfadjointView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2,2).template selfadjointView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template selfadjointView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2,2).template selfadjointView<Eigen::Lower>()))> == 1);

  constant_adapter<M33, double, 5> c533 {};
  constant_adapter<M3x, double, 5> c53x_3 {3};
  constant_adapter<Mx3, double, 5> c5x3_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_33 {3, 3};

  constant_adapter<M31, double, 5> c531 {};
  constant_adapter<M3x, double, 5> c53x_1 {1};
  constant_adapter<Mx1, double, 5> c5x1_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_31 {3, 1};

  auto m33_d5 = make_dense_object_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5);

  EXPECT_TRUE(is_near(to_diagonal(c531), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c53x_1), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c5x1_3), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c5xx_31), m33_d5));

  EXPECT_TRUE(is_near(diagonal_of(c533), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c53x_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c5x3_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c5xx_33), M31::Constant(5)));
  static_assert(constant_value_v<decltype(diagonal_of(c5xx_33))> == 5);

  ZA21 z21;
  ZA20 z2x_1 {1};
  ZA01 zx1_2 {2};
  ZA00 zxx_21 {2, 1};

  ZA12 z12;
  ZA10 z1x_2 {2};
  ZA02 zx2_1 {1};
  ZA00 zxx_12 {1, 2};

  EXPECT_TRUE(is_near(to_diagonal(z21), z22));
  EXPECT_TRUE(is_near(to_diagonal(z2x_1), z22));
  EXPECT_TRUE(is_near(to_diagonal(zx1_2), z22));
  EXPECT_TRUE(is_near(to_diagonal(zxx_21), z22));
  static_assert(diagonal_matrix<decltype(to_diagonal(z21))> and internal::has_nested_vector<decltype(to_diagonal(z21))>);
  static_assert(diagonal_matrix<decltype(to_diagonal(zx1_2))> and internal::has_nested_vector<decltype(to_diagonal(zx1_2))>);
  static_assert(diagonal_matrix<decltype(to_diagonal(z2x_1))> and internal::has_nested_vector<decltype(to_diagonal(z2x_1))>);
  static_assert(diagonal_matrix<decltype(to_diagonal(zxx_21))> and internal::has_nested_vector<decltype(to_diagonal(zxx_21))>);
  static_assert(zero<decltype(to_diagonal(zxx_21))>);

  EXPECT_TRUE(is_near(diagonal_of(z22), z21));
  EXPECT_TRUE(is_near(diagonal_of(z2x_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(zx2_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(zxx_22), z21));
  static_assert(zero<decltype(diagonal_of(zxx_22))>);
}


TEST(adapters, constant_elementwise)
{
  auto m23 = make_dense_object_from<M23>(5.5, 5.5, 5.5, 5.5, 5.5, 5.5);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions{2}, Dimensions<3>{}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions<2>{}, Dimensions{3}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions{2}, Dimensions{3}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23>([]{return 5.5;}), m23));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero<M3x>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero<Mx3>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero<Mxx>(3,3)), M33::Constant(1)));
}


TEST(adapters, constant_reductions)
{
  // reduce zero

  auto ez11 = M11::Identity() - M11::Identity();
  using EZ11 = decltype(ez11);

  auto ez23 = Eigen::Replicate<EZ11, 2, 3> {ez11};
  auto ez2x_3 = Eigen::Replicate<EZ11, 2, Eigen::Dynamic> {ez11, 2, 3};
  auto ezx3_2 = Eigen::Replicate<EZ11, Eigen::Dynamic, 3> {ez11, 2, 3};
  auto ezxx_23 = Eigen::Replicate<EZ11, Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 3};

  auto ez13 = Eigen::Replicate<EZ11, 1, 3> {ez11};
  auto ez21 = Eigen::Replicate<EZ11, 2, 1> {ez11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ez23), ez13)); static_assert(zero<decltype(reduce<0>(std::plus<double>{}, ez23))>);
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ezx3_2), ez13)); static_assert(zero<decltype(reduce<0>(std::plus<double>{}, ezx3_2))>);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ez2x_3), ez13)); static_assert(zero<decltype(reduce<0>(std::plus<double>{}, ez2x_3))>);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ezxx_23), ez13)); static_assert(zero<decltype(reduce<0>(std::plus<double>{}, ezxx_23))>);

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ez23), ez21)); static_assert(zero<decltype(reduce<1>(std::plus<double>{}, ez23))>);
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ezx3_2), ez21)); static_assert(zero<decltype(reduce<1>(std::plus<double>{}, ezx3_2))>);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ez2x_3), ez21)); static_assert(zero<decltype(reduce<1>(std::plus<double>{}, ez2x_3))>);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ezxx_23), ez21)); static_assert(zero<decltype(reduce<1>(std::plus<double>{}, ezxx_23))>);

  EXPECT_EQ((reduce(std::plus<double>{}, ez23)), 0);
  EXPECT_EQ((reduce(std::plus<double>{}, ezx3_2)), 0);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, ez23), ez11));
  EXPECT_EQ((reduce(std::multiplies<double>{}, ezxx_23)), 0);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, ez2x_3), ez11));

  static_assert(reduce(std::plus<double>{}, ez11) == 0);
  static_assert(reduce(std::plus<double>{}, ez13) == 0);
  static_assert(reduce(std::plus<double>{}, ez21) == 0);
  static_assert(reduce(std::plus<double>{}, ez23) == 0);
  static_assert(reduce(std::multiplies<double>{}, ez23) == 0);

  // reduce constant

  auto ec11 = M11::Identity() + M11::Identity();
  using EC11 = decltype(ec11);

  auto ec23 = Eigen::Replicate<EC11, 2, 3> {ec11};
  auto ec2x_3 = Eigen::Replicate<EC11, 2, Eigen::Dynamic> {ec11, 2, 3};
  auto ecx3_2 = Eigen::Replicate<EC11, Eigen::Dynamic, 3> {ec11, 2, 3};
  auto ecxx_23 = Eigen::Replicate<EC11, Eigen::Dynamic, Eigen::Dynamic> {ec11, 2, 3};

  auto ec13 = Eigen::Replicate<EC11, 1, 3> {ec11};
  auto ec21 = Eigen::Replicate<EC11, 2, 1> {ec11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ec23), 2 * ec13));
  static_assert(constant_value_v<decltype(reduce<0>(std::plus<double>{}, ec23))> == 4);
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, ecx3_2), 2 * ec13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ec2x_3), 2 * ec13));
  static_assert(constant_value_v<decltype(reduce<0>(std::plus<double>{}, ec2x_3))> == 4);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, ecxx_23), 2 * ec13));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ec23), 3 * ec21));
  static_assert(constant_value_v<decltype(reduce<1>(std::plus<double>{}, ec23))> == 6);
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, ecx3_2), 3 * ec21));
  static_assert(constant_value_v<decltype(reduce<1>(std::plus<double>{}, ecx3_2))> == 6);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ec2x_3), 4 * ec21));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, ecxx_23), 4 * ec21));

  EXPECT_EQ((reduce(std::plus<double>{}, ec23)), 12);
  EXPECT_EQ((reduce(std::plus<double>{}, ecx3_2)), 12);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, ec23), 6 * ec11));
  EXPECT_EQ((reduce(std::multiplies<double>{}, ecxx_23)), 64);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, ec2x_3), 32 * ec11));

  double non_constexpr_dummy = 0.0;
  EXPECT_TRUE(is_near(reduce<1, 0>([&](auto a, auto b){ return non_constexpr_dummy + a + b; }, ec23), 6 * ec11));

#if defined(__cpp_lib_ranges) and not defined (__clang__)
  static_assert(reduce(std::plus<double>{}, ec11) == 2);
  static_assert(reduce(std::multiplies<double>{}, ec11) == 2);
#else
  EXPECT_EQ(reduce(std::plus<double>{}, ec11), 2);
  EXPECT_EQ(reduce(std::multiplies<double>{}, ec11), 2);
#endif
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
  static_assert(values::internal::near(constant_value_v<EFC11>, 2./3));

  auto efc23 = Eigen::Replicate<EFC11, 2, 3> {efc11};
  auto efc2x_3 = Eigen::Replicate<EFC11, 2, Eigen::Dynamic> {efc11, 2, 3};
  auto efcx3_2 = Eigen::Replicate<EFC11, Eigen::Dynamic, 3> {efc11, 2, 3};
  auto efcxx_23 = Eigen::Replicate<EFC11, Eigen::Dynamic, Eigen::Dynamic> {efc11, 2, 3};

  auto efc13 = Eigen::Replicate<EFC11, 1, 3> {efc11};
  auto efc21 = Eigen::Replicate<EFC11, 2, 1> {efc11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, efc23), 2 * efc13));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, efcx3_2), 2 * efc13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, efc2x_3), 2./3 * efc13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, efcxx_23), 2./3 * efc13));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, efc23), 3 * efc21)); static_assert(values::internal::near(constant_value_v<decltype(reduce<1>(std::plus<double>{}, efc23))>, 2));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, efcx3_2), 3 * efc21)); static_assert(values::internal::near(constant_value_v<decltype(reduce<1>(std::plus<double>{}, efcx3_2))>, 2));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, efc2x_3), 4./9 * efc21));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, efcxx_23), 4./9 * efc21));

  EXPECT_EQ((reduce(std::plus<double>{}, efc23)), 4);
  EXPECT_EQ((reduce(std::plus<double>{}, efcx3_2)), 4);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, efc23), 6 * efc11));
  EXPECT_NEAR((reduce(std::multiplies<double>{}, efcxx_23)), 64./729, 1e-9);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, efc2x_3), 32./243 * efc11));

#if defined(__cpp_lib_ranges) and not defined (__clang__)
  static_assert(values::internal::near(reduce(std::plus<double>{}, efc11), 2./3));
  static_assert(values::internal::near(reduce(std::multiplies<double>{}, efc11), 2./3));
#else
  EXPECT_NEAR(reduce(std::plus<double>{}, efc11), 2./3, 1e-9);
  EXPECT_NEAR(reduce(std::multiplies<double>{}, efc11), 2./3, 1e-9);
#endif
  static_assert(values::internal::near(reduce(std::plus<double>{}, efc13), 2));
  static_assert(values::internal::near(reduce(std::multiplies<double>{}, efc13), 8./27));
  static_assert(values::internal::near(reduce(std::plus<double>{}, efc21), 4./3));
  static_assert(values::internal::near(reduce(std::multiplies<double>{}, efc21), 4./9));
  static_assert(values::internal::near(reduce(std::plus<double>{}, efc23), 4));
  static_assert(values::internal::near(reduce(std::multiplies<double>{}, efc23), 64./729));

  // average_reduce zero

  auto ez22 = M22::Identity() - M22::Identity();
  auto ez2x_2 = Eigen::Replicate<decltype(ez11), 2, Eigen::Dynamic> {ez11, 2, 2};
  auto ezx2_2 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, 2> {ez11, 2, 2};
  auto ezxx_22 = Eigen::Replicate<decltype(ez11), Eigen::Dynamic, Eigen::Dynamic> {ez11, 2, 2};

  EXPECT_TRUE(is_near(average_reduce<1>(ez22), ez21)); static_assert(zero<decltype(average_reduce<1>(ez22))>);
  EXPECT_TRUE(is_near(average_reduce<1>(ez2x_2), ez21)); static_assert(zero<decltype(average_reduce<1>(ez2x_2))>);
  EXPECT_TRUE(is_near(average_reduce<1>(ezx2_2), ez21)); static_assert(zero<decltype(average_reduce<1>(ezx2_2))>);
  EXPECT_TRUE(is_near(average_reduce<1>(ezxx_22), ez21)); static_assert(zero<decltype(average_reduce<1>(ezxx_22))>);

  auto ez12 = Eigen::Replicate<decltype(ez11), 1, 2> {ez11, 1, 2};

  EXPECT_TRUE(is_near(average_reduce<0>(ez22), ez12)); static_assert(zero<decltype(average_reduce<0>(ez22))>);
  EXPECT_TRUE(is_near(average_reduce<0>(ez2x_2), ez12)); static_assert(zero<decltype(average_reduce<0>(ez2x_2))>);
  EXPECT_TRUE(is_near(average_reduce<0>(ezx2_2), ez12)); static_assert(zero<decltype(average_reduce<0>(ezx2_2))>);
  EXPECT_TRUE(is_near(average_reduce<0>(ezxx_22), ez12)); static_assert(zero<decltype(average_reduce<0>(ezxx_22))>);

  static_assert(average_reduce(ez22) == 0);
  static_assert(average_reduce(ez2x_2) == 0);
  static_assert(average_reduce(ezx2_2) == 0);
  static_assert(average_reduce(ezxx_22) == 0);

  // average_reduce constant

  auto i22 = M22::Identity();
  auto i2x_2 = M2x::Identity(2, 2);
  auto ix2_2 = Mx2::Identity(2, 2);
  auto ixx_22 = Mxx::Identity(2, 2);

  EXPECT_TRUE(is_near(average_reduce<1>(i22), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i2x_2), M21::Constant(0.5)));
  auto rcix2_2 = average_reduce<1>(ix2_2);
  EXPECT_TRUE(is_near(rcix2_2, M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(ix2_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(ixx_22), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(average_reduce<1>(M2x::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(Mx2::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(Mxx::Identity(2, 2)), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(average_reduce<1>(ec23), M21::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<1>(ec23))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(ec2x_3), M21::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<1>(ec2x_3))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(ecx3_2), M21::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<1>(ecx3_2))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(ecxx_23), M21::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<1>(ecxx_23))> == 2);

  EXPECT_TRUE(is_near(average_reduce<0>(ec23), M13::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<0>(ec23))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(ec2x_3), M13::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<0>(ec2x_3))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(ecx3_2), M13::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<0>(ecx3_2))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(ecxx_23), M13::Constant(2))); static_assert(constant_value_v<decltype(average_reduce<0>(ecxx_23))> == 2);

  static_assert(average_reduce(ec23) == 2);
  static_assert(average_reduce(ec2x_3) == 2);
  static_assert(average_reduce(ecx3_2) == 2);
  static_assert(average_reduce(ecxx_23) == 2);

  // average_reduce constant diagonal

  auto ec11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = ec11_2.replicate<2, 1>().asDiagonal();
  auto d2x_1_2 = Eigen::Replicate<decltype(ec11_2), 2, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();
  auto dx1_2_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, 1> {ec11_2, 2, 1}.asDiagonal();
  auto dxx_21_2 = Eigen::Replicate<decltype(ec11_2), Eigen::Dynamic, Eigen::Dynamic> {ec11_2, 2, 1}.asDiagonal();

  static_assert(constant_value_v<decltype(average_reduce<1>(d21_2))> == 1);
  EXPECT_TRUE(is_near(average_reduce<1>(d2x_1_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<1>(dx1_2_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<1>(dxx_21_2), M21::Constant(1)));

  static_assert(constant_value_v<decltype(average_reduce<0>(d21_2))> == 1);
  EXPECT_TRUE(is_near(average_reduce<0>(d2x_1_2), M12::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<0>(dx1_2_2), M12::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<0>(dxx_21_2), M12::Constant(1)));

  static_assert(constant_value_v<decltype(average_reduce<0>(d21_2))> == 1);
  static_assert(constant_value_v<decltype(average_reduce<1>(d21_2))> == 1);

  static_assert(average_reduce(d21_2) == 1);
  EXPECT_EQ(average_reduce(d2x_1_2), 1);
  EXPECT_EQ(average_reduce(dx1_2_2), 1);
  EXPECT_EQ(average_reduce(dxx_21_2), 1);

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

  // reduce zero_adapter

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z2x_2 {Dimensions<2>(), 2};
  ZA02 zx2_2 {2, Dimensions<2>()};
  ZA00 zxx_22 {2, 2};

  EXPECT_TRUE(is_near(average_reduce<1>(z22), ez21));
  EXPECT_TRUE(is_near(average_reduce<1>(z2x_2), ez21));
  EXPECT_TRUE(is_near(average_reduce<1>(zx2_2), ez21));
  EXPECT_TRUE(is_near(average_reduce<1>(zxx_22), ez21));
  static_assert(zero<decltype(average_reduce<1>(zxx_22))>);

  EXPECT_TRUE(is_near(average_reduce<0>(z22), ez12));
  EXPECT_TRUE(is_near(average_reduce<0>(z2x_2), ez12));
  EXPECT_TRUE(is_near(average_reduce<0>(zx2_2), ez12));
  EXPECT_TRUE(is_near(average_reduce<0>(zxx_22), ez12));
  static_assert(zero<decltype(average_reduce<0>(zxx_22))>);

  // reduce constant_adapter

  constant_adapter<M34, double, 5> c534 {};
  constant_adapter<M3x, double, 5> c53x_4 {4};
  constant_adapter<Mx4, double, 5> c5x4_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_34 {3, 4};

  constant_adapter<M33, double, 5> c533 {};
  constant_adapter<M3x, double, 5> c53x_3 {3};
  constant_adapter<Mx3, double, 5> c5x3_3 {3};
  constant_adapter<Mxx, double, 5> c5xx_33 {3, 3};

  auto colzc34 = average_reduce<1>(c5xx_34);
  EXPECT_TRUE(is_near(average_reduce<1>(constant_adapter<M23, double, 3> ()), (M21::Constant(3))));
  EXPECT_EQ(colzc34, (constant_adapter<M31, double, 5> {}));
  EXPECT_EQ(get_pattern_collection<0>(colzc34), 3);
  EXPECT_EQ(get_pattern_collection<1>(colzc34), 1);
  static_assert(constant_matrix<decltype(colzc34)>);

  auto rowzc34 = average_reduce<0>(c5xx_34);
  EXPECT_TRUE(is_near(average_reduce<0>(constant_adapter<M23, double, 3> ()), (M13::Constant(3))));
  EXPECT_EQ(rowzc34, (constant_adapter<eigen_matrix_t<double, 1, 4>, double, 5> {}));
  EXPECT_EQ(get_pattern_collection<1>(rowzc34), 4);
  EXPECT_EQ(get_pattern_collection<0>(rowzc34), 1);
  static_assert(constant_matrix<decltype(rowzc34)>);
}


TEST(adapters, constant_element_functions)
{
  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z2x_2 {Dimensions<2>(), 2};
  ZA02 zx2_2 {2, Dimensions<2>()};
  ZA00 zxx_22 {2, 2};

  ZA21 z21;
  ZA20 z2x_1 {1};
  ZA01 zx1_2 {2};
  ZA00 zxx_21 {2, 1};

  ZA12 z12;
  ZA10 z1x_2 {2};
  ZA02 zx2_1 {1};
  ZA00 zxx_12 {1, 2};

  // get_component
  EXPECT_NEAR(get_component(z22, 1, 0), 0, 1e-8);
  EXPECT_NEAR(get_component(z2x_2, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_component(zx2_2, 0, 1), 0, 1e-6);
  EXPECT_NEAR(get_component(zxx_22, 1, 0), 0, 1e-6);

  EXPECT_NEAR(get_component(z21, 0, 0), 0, 1e-8);
  EXPECT_NEAR(get_component(z2x_1, 1, 0), 0, 1e-6);
  EXPECT_NEAR(get_component(zx1_2, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_component(zxx_21, 1, 0), 0, 1e-6);

  EXPECT_NEAR(get_component(z21, 0), 0, 1e-8);
  EXPECT_NEAR(get_component(zx1_2, 0), 0, 1e-6);

  EXPECT_NEAR(get_component(z12, 0, 0), 0, 1e-8);
  EXPECT_NEAR(get_component(z1x_2, 0, 1), 0, 1e-6);
  EXPECT_NEAR(get_component(zx2_1, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_component(zxx_12, 0, 1), 0, 1e-6);

  EXPECT_NEAR(get_component(constant_adapter<M22, double, 5> {}, 1, 0), 5, 1e-8);

  constant_adapter<Mxx, double, 5> cxx {2, 2};

  EXPECT_NEAR((get_component(cxx, 0, 0)), 5, 1e-6);
  EXPECT_NEAR((get_component(cxx, 0, 1)), 5, 1e-6);
  EXPECT_NEAR((get_component(cxx, 1, 0)), 5, 1e-6);
  EXPECT_NEAR((get_component(cxx, 1, 1)), 5, 1e-6);
  EXPECT_NEAR((get_component(constant_adapter<Mx1, double, 7> {3}, 0)), 7, 1e-6);

  // get_slice

  std::integral_constant<std::size_t, 0> N0;
  std::integral_constant<std::size_t, 1> N1;
  std::integral_constant<std::size_t, 2> N2;
  std::integral_constant<std::size_t, 3> N3;

  auto z34 = make_zero<M34>();

  EXPECT_TRUE(is_near(get_slice(z34, std::tuple{N0, N0}, std::tuple{N2, N2}), make_zero<M22>()));
  EXPECT_TRUE(is_near(get_slice(z34, std::tuple{N1, N1}, std::tuple{2, N3}), make_zero<M23>()));
  EXPECT_TRUE(is_near(get_slice(z34, std::tuple{0, 0}, std::tuple{2, N2}), make_zero<M22>()));
  EXPECT_TRUE(is_near(get_slice(z34, std::tuple{0, 0}, std::tuple{2, N2}), make_zero<M22>()));
  EXPECT_TRUE(is_near(get_slice<1, 0>(z34, std::tuple{N0, N0}, std::tuple{N3, N2}), make_zero<M23>()));
  EXPECT_TRUE(is_near(get_slice<0>(z34, std::tuple{N0}, std::tuple{N2}), make_zero<M24>()));
  EXPECT_TRUE(is_near(get_slice<0>(z34, std::tuple{1}, std::tuple{1}), make_zero<M14>()));
  EXPECT_TRUE(is_near(get_slice<1>(z34, std::tuple{N0}, std::tuple{N2}), make_zero<M32>()));
  EXPECT_TRUE(is_near(get_slice<1>(z34, std::tuple{1}, std::tuple{1}), make_zero<M31>()));

  auto c34 = make_constant<M34, double, 3>();

  EXPECT_TRUE(is_near(get_slice(c34, std::tuple{N0, N0}, std::tuple{N2, N2}), make_constant<M22, double, 3>()));
  EXPECT_TRUE(is_near(get_slice(c34, std::tuple{N1, N1}, std::tuple{2, N3}), make_constant<M23, double, 3>()));
  EXPECT_TRUE(is_near(get_slice(c34, std::tuple{0, 0}, std::tuple{2, N2}), make_constant<M22, double, 3>()));
  EXPECT_TRUE(is_near(get_slice(c34, std::tuple{0, 0}, std::tuple{2, N2}), make_constant<M22, double, 3>()));
  EXPECT_TRUE(is_near(get_slice<1, 0>(c34, std::tuple{N0, N0}, std::tuple{N3, N2}), make_constant<M23, double, 3>()));
  EXPECT_TRUE(is_near(get_slice<0>(c34, std::tuple{N0}, std::tuple{N2}), make_constant<M24, double, 3>()));
  EXPECT_TRUE(is_near(get_slice<0>(c34, std::tuple{1}, std::tuple{1}), make_constant<M14, double, 3>()));
  EXPECT_TRUE(is_near(get_slice<1>(c34, std::tuple{N0}, std::tuple{N2}), make_constant<M32, double, 3>()));
  EXPECT_TRUE(is_near(get_slice<1>(c34, std::tuple{1}, std::tuple{1}), make_constant<M31, double, 3>()));

  auto c34r = make_constant<M34>(3.);

  EXPECT_TRUE(is_near(get_slice(c34r, std::tuple{N0, N0}, std::tuple{N2, N2}), make_constant<M22>(3.)));
  EXPECT_TRUE(is_near(get_slice(c34r, std::tuple{N1, N1}, std::tuple{2, N3}), make_constant<M23>(3.)));
  EXPECT_TRUE(is_near(get_slice(c34r, std::tuple{0, 0}, std::tuple{2, N2}), make_constant<M22>(3.)));
  EXPECT_TRUE(is_near(get_slice(c34r, std::tuple{0, 0}, std::tuple{2, N2}), make_constant<M22>(3.)));
  EXPECT_TRUE(is_near(get_slice<1, 0>(c34r, std::tuple{N0, N0}, std::tuple{N3, N2}), make_constant<M23>(3.)));
  EXPECT_TRUE(is_near(get_slice<0>(c34r, std::tuple{N0}, std::tuple{N2}), make_constant<M24>(3.)));
  EXPECT_TRUE(is_near(get_slice<0>(c34r, std::tuple{1}, std::tuple{1}), make_constant<M14>(3.)));
  EXPECT_TRUE(is_near(get_slice<1>(c34r, std::tuple{N0}, std::tuple{N2}), make_constant<M32>(3.)));
  EXPECT_TRUE(is_near(get_slice<1>(c34r, std::tuple{1}, std::tuple{1}), make_constant<M31>(3.)));

  // get_chip

  constant_adapter<Mxx, double, 5> c5xx_34 {3, 4};

  EXPECT_TRUE(is_near(get_chip<1>(constant_adapter<M23, double, 6> {}, 1), (M21::Constant(6))));
  EXPECT_TRUE(is_near(get_chip<1>(constant_adapter<M23, double, 7> {}, N1), (M21::Constant(7))));

  auto c5c34 = get_chip<1>(c5xx_34, 1);
  EXPECT_EQ(get_pattern_collection<0>(c5c34), 3);
  static_assert(dimension_size_of_index_is<decltype(c5c34), 1, 1>);
  static_assert(constant_value_v<decltype(c5c34)> == 5);

  auto c5v34 = get_chip<1>(constant_adapter<Mx4, double, 5> {3}, N1);
  EXPECT_EQ(get_pattern_collection<0>(c5v34), 3);
  static_assert(dimension_size_of_index_is<decltype(c5v34), 1, 1>);
  static_assert(constant_value_v<decltype(c5v34)> == 5);

  EXPECT_TRUE(is_near(get_chip<0>(constant_adapter<M32, double, 6> {}, 1), (M12::Constant(6))));
  EXPECT_TRUE(is_near(get_chip<0>(constant_adapter<M32, double, 7> {}, N1), (M12::Constant(7))));

  auto r5c34 = get_chip<0>(c5xx_34, 1);
  EXPECT_EQ(get_pattern_collection<1>(r5c34), 4);
  static_assert(dimension_size_of_index_is<decltype(r5c34), 0, 1>);
  static_assert(constant_value_v<decltype(r5c34)> == 5);

  auto r5v34 = get_chip<0>(constant_adapter<M3x, double, 5> {4}, N1);
  EXPECT_EQ(get_pattern_collection<1>(r5v34), 4);
  static_assert(dimension_size_of_index_is<decltype(r5v34), 0, 1>);
  static_assert(constant_value_v<decltype(r5v34)> == 5);

  EXPECT_TRUE(is_near(get_chip<1>(zero_adapter<M23>(), N1), (M21::Zero())));
  EXPECT_TRUE(is_near(get_chip<1>(zero_adapter<M23>(), 1), (M21::Zero())));

  auto zc34 = zero_adapter<M34> {};
  auto czc34 = get_chip<1>(zc34, 1);
  EXPECT_EQ(get_pattern_collection<0>(czc34), 3);
  static_assert(dimension_size_of_index_is<decltype(czc34), 1, 1>);
  static_assert(zero<decltype(czc34)>);

  auto czv34 = get_chip<1>(zero_adapter<Mx4> {3}, N1);
  EXPECT_EQ(get_pattern_collection<0>(czv34), 3);
  static_assert(dimension_size_of_index_is<decltype(czv34), 1, 1>);
  static_assert(zero<decltype(czv34)>);

  EXPECT_TRUE(is_near(get_chip<1>(make_zero<M33>(Dimensions<3>{}, Dimensions<3>{}), N1), make_dense_object_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<1>(make_zero<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_dense_object_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<1>(make_zero<M3x>(Dimensions<3>{}, 3), 2), make_dense_object_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<1>(make_zero<Mx3>(3, Dimensions<3>{}), 2), make_dense_object_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<1>(make_zero<Mxx>(3,3), 2), make_dense_object_from<M31>(0., 0, 0)));

  EXPECT_TRUE(is_near(get_chip<0>(zero_adapter<M32>(Dimensions<3>{}, Dimensions<2>{}), N1), (M12::Zero())));
  EXPECT_TRUE(is_near(get_chip<0>(zero_adapter<M32>(Dimensions<3>{}, Dimensions<2>{}), 1), (M12::Zero())));

  auto rzc34 = get_chip<0>(zc34, 1);
  EXPECT_EQ(get_pattern_collection<1>(rzc34), 4);
  static_assert(dimension_size_of_index_is<decltype(rzc34), 0, 1>);
  static_assert(zero<decltype(rzc34)>);

  auto rzv34 = get_chip<0>(zero_adapter<M3x> {Dimensions<3>{}, 4}, N1);
  EXPECT_EQ(get_pattern_collection<1>(rzv34), 4);
  static_assert(dimension_size_of_index_is<decltype(rzv34), 0, 1>);
  static_assert(zero<decltype(rzv34)>);

  EXPECT_TRUE(is_near(get_chip<0>(make_zero<M33>(Dimensions<3>{}, Dimensions<3>{}), N1), make_dense_object_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<0>(make_zero<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_dense_object_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<0>(make_zero<M3x>(Dimensions<3>{}, 3), 2), make_dense_object_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<0>(make_zero<Mx3>(3, Dimensions<3>{}), 2), make_dense_object_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_chip<0>(make_zero<Mxx>(3,3), 2), make_dense_object_from<M13>(0., 0, 0)));

  // \todo tile
}


TEST(adapters, constant_concatenate)
{
  // General diagonal concatenate relies on constructing zero matrices.
  auto m23 = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto m22 = make_dense_object_from<M22>(7, 8, 9, 10);
  auto m45 = make_dense_object_from<M45>(
    1, 2, 3, 0, 0,
    4, 5, 6, 0, 0,
    0, 0, 0, 7, 8,
    0, 0, 0, 9, 10);

  EXPECT_TRUE(is_near(concatenate<0, 1>(m23, m22), m45));
  EXPECT_TRUE(is_near(concatenate<1, 0>(M2x {m23}, m22), m45));
  EXPECT_TRUE(is_near(concatenate<0, 1>(M2x {m23}, M2x {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<1, 0>(M2x {m23}, Mx2 {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<0, 1>(M2x {m23}, Mxx {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<1, 0>(Mx3 {m23}, m22), m45));
  EXPECT_TRUE(is_near(concatenate<0, 1>(Mx3 {m23}, M2x {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<1, 0>(Mx3 {m23}, Mx2 {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<0, 1>(Mx3 {m23}, Mxx {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<1, 0>(Mxx {m23}, m22), m45));
  EXPECT_TRUE(is_near(concatenate<0, 1>(Mxx {m23}, M2x {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<1, 0>(Mxx {m23}, Mx2 {m22}), m45));
  EXPECT_TRUE(is_near(concatenate<0, 1>(Mxx {m23}, Mxx {m22}), m45));
}


TEST(adapters, constant_split)
{
  auto tup_z33_z23 = std::tuple {make_zero<M33>(), make_zero<M23>()};

  EXPECT_TRUE(is_near(split<0>(make_zero<Mxx>(Dimensions<5>{}, Dimensions<3>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero<Mxx>(Dimensions<5>{}, 3), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero<Mxx>(5, Dimensions<3>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero<Mxx>(5, 3), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));

  auto tup_z33_z32 = std::tuple {make_zero<M33>(), make_zero<M32>()};

  EXPECT_TRUE(is_near(split<1>(make_zero<Mxx>(Dimensions<3>{}, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero<Mxx>(Dimensions<3>{}, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero<Mxx>(3, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero<Mxx>(3, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));

  auto tup_z33_z22 = std::tuple {make_zero<M33>(), make_zero<M22>()};

  EXPECT_TRUE(is_near(split<0, 1>(make_zero<M55>(Dimensions<5>{}, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero<M5x>(Dimensions<5>{}, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero<Mx5>(5, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero<Mxx>(5, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));

  auto tup_z32_z23 = std::tuple {make_zero<M32>(), make_zero<M23>()};

  EXPECT_TRUE(is_near(split<0, 1>(make_zero<M55>(Dimensions<5>{}, Dimensions<5>{}), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero<M5x>(Dimensions<5>{}, 5), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero<Mx5>(5, Dimensions<5>{}), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero<Mxx>(5, 5), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
}


TEST(adapters, constant_chipwise_operations)
{
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + make_constant<double, 1>(row)); }, make_zero<M3x>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero<Mx3>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + make_constant<double, 1>(row)); }, make_zero<Mxx>(3,3)), M33::Constant(1)));

  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero<M3x>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + make_constant<double, 1>(col)); }, make_zero<Mx3>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + make_constant<double, 1>(col)); }, make_zero<Mxx>(3,3)), M33::Constant(1)));
}


TEST(adapters, sum)
{
  // zero

  static_assert(zero<decltype(sum(std::declval<C22_2>(), std::declval<Z22>(), std::declval<C22_m2>()))>);

  // constant

  static_assert(constant_value_v<decltype(sum(std::declval<C22_m2>(), std::declval<C22_m2>()))> == -4);
  static_assert(constant_value_v<decltype(sum(std::declval<C22_1>(), std::declval<Z22>(), std::declval<C22_m2>()))> == -1);
  static_assert(constant_value_v<decltype(sum(std::declval<Z22>(), std::declval<Z22>(), std::declval<C22_1>(), std::declval<Z22>(), std::declval<C22_m2>()))> == -1);

  // constant diagonal

  static_assert(constant_diagonal_value_v<decltype(sum(std::declval<Cd22_2>(), std::declval<Cd22_3>()))> == 5);
  static_assert(constant_diagonal_value_v<decltype(sum(std::declval<Z22>(), std::declval<Cd22_m2>(), std::declval<Cd22_3>()))> == 1);
}

