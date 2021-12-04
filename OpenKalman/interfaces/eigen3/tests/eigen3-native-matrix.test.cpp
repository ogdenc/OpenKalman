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

  using M00 = eigen_matrix_t<double, 0, 0>;
  using M10 = eigen_matrix_t<double, 1, 0>;
  using M01 = eigen_matrix_t<double, 0, 1>;
  using M20 = eigen_matrix_t<double, 2, 0>;
  using M02 = eigen_matrix_t<double, 0, 2>;
  using M30 = eigen_matrix_t<double, 3, 0>;
  using M03 = eigen_matrix_t<double, 0, 3>;
  using M04 = eigen_matrix_t<double, 0, 4>;
  using M40 = eigen_matrix_t<double, 4, 0>;
  using M50 = eigen_matrix_t<double, 5, 0>;
  using M05 = eigen_matrix_t<double, 0, 5>;

  using cdouble = std::complex<double>;

  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, 0>;
  using CM02 = eigen_matrix_t<cdouble, 0, 2>;
  using CM00 = eigen_matrix_t<cdouble, 0, 0>;

  using Axis2 = Coefficients<Axis, Axis>;

  auto z11 = M11::Identity() - M11::Identity(); using Z11 = decltype(z11);
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
  auto c11_3 = c11_2 + c11_1; using C11_3 = decltype(c11_3);

  using C21_1 = Eigen::Replicate<C11_1, 2, 1>;
  using C20_1 = Eigen::Replicate<C11_1, 2, Eigen::Dynamic>;
  using C01_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 1>;
  using C00_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, Eigen::Dynamic>;

  auto c21_1 = C21_1 {c11_1, 2, 1};
  auto c20_1_1 = C20_1 {c11_1, 2, 1};
  auto c01_2_1 = C01_1 {c11_1, 2, 1};
  auto c00_21_1 = C00_1 {c11_1, 2, 1};

  using C12_1 = Eigen::Replicate<C11_1, 1, 2>;
  using C02_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 2>;
  using C10_1 = Eigen::Replicate<C11_1, 1, Eigen::Dynamic>;

  auto c12_1 = C12_1 {c11_1, 2, 1};
  auto c10_2_1 = C10_1 {c11_1, 2, 1};
  auto c02_1_1 = C02_1 {c11_1, 2, 1};
  auto c00_12_1 = C00_1 {c11_1, 2, 1};

  auto c21_m2 = c11_m2.replicate<2,1>(); using C21_m2 = decltype(c21_m2);

  using C22_2 = Eigen::Replicate<C11_2, 2, 2>;
  using C21_2 = Eigen::Replicate<C11_2, 2, 1>;
  using C20_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using C02_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C00_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;
  using C01_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 1>;

  auto c22_2 = C22_2 {c11_2, 2, 2};
  auto c20_2_2 = C20_2 {c11_2, 2, 2};
  auto c02_2_2 = C02_2 {c11_2, 2, 2};
  auto c00_22_2 = C00_2 {c11_2, 2, 2};

  auto c22_1 = c11_1.replicate<2,2>(); using C22_1 = decltype(c22_1);
  auto c22_m2 = c11_m2.replicate<2,2>(); using C22_m2 = decltype(c22_m2);

  auto c21_2 = C21_2 {c11_2, 2, 1};
  auto c20_1_2 = C20_2 {c11_2, 2, 1};
  auto c01_2_2 = C01_2 {c11_2, 2, 1};
  auto c00_21_2 = C00_2 {c11_2, 2, 1};

  auto c12_2 = c11_2.replicate<1,2>(); using C12_2 = decltype(c12_2);

  using C21_3 = Eigen::Replicate<C11_3, 2, 1>;
  using C20_3 = Eigen::Replicate<C11_3, 2, Eigen::Dynamic>;
  using C01_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, 1>;
  using C00_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, Eigen::Dynamic>;

  auto c21_3 = C21_3 {c11_3, 2, 1};
  auto c20_1_3 = C20_3 {c11_3, 2, 1};
  auto c01_2_3 = C01_3 {c11_3, 2, 1};
  auto c00_21_3 = C00_3 {c11_3, 2, 1};

  auto b11_true = eigen_matrix_t<bool, 1, 1>::Identity(); using B11_true = decltype(b11_true);
  auto b11_false = (not b11_true.array()).matrix(); using B11_false = decltype(b11_false);

  auto b22_true = b11_true.replicate<2,2>(); using B22_true = decltype(b22_true);
  auto b22_false = b11_false.replicate<2,2>(); using B22_false = decltype(b22_false);

  auto i22 = M22::Identity(); using I22 = decltype(i22);
  auto i20_2 = c20_1_1.asDiagonal(); using I20_2 = decltype(i20_2);
  auto i02_2 = c01_2_1.asDiagonal(); using I02_2 = decltype(i02_2);
  auto i00_22 = c00_21_1.asDiagonal(); using I00_22 = decltype(i00_22);

  auto d22_2 = c21_2.asDiagonal(); using D22_2 = decltype(d22_2);
  auto d20_2_2 = c20_1_2.asDiagonal(); using D20_2_2 = decltype(d20_2_2);
  auto d02_2_2 = c01_2_2.asDiagonal(); using D02_2_2 = decltype(d02_2_2);
  auto d00_22_2 = c00_21_2.asDiagonal(); using D00_22_2 = decltype(d00_22_2);

  auto d22_3 = c21_3.asDiagonal(); using D22_3 = decltype(d22_3);
  auto d20_2_3 = c20_1_3.asDiagonal(); using D20_2_3 = decltype(d20_2_3);
  auto d02_2_3 = c01_2_3.asDiagonal(); using D02_2_3 = decltype(d02_2_3);
  auto d00_22_3 = c00_21_3.asDiagonal(); using D00_22_3 = decltype(d00_22_3);
}


TEST(eigen3, Eigen_Matrix_class_traits)
{
  static_assert(native_eigen_matrix<M22>);
  static_assert(not native_eigen_matrix<double>);

  static_assert(eigen_matrix<M22>);
  static_assert(not eigen_matrix<double>);

  static_assert(not constant_matrix<decltype(M00::Zero(2, 3))>); // because the constant is not known at compile time
  static_assert(constant_coefficient_v<Z11> == 0);
  static_assert(constant_coefficient_v<Z22> == 0);
  static_assert(constant_coefficient_v<Z21> == 0);
  static_assert(constant_coefficient_v<Z23> == 0);
  static_assert(constant_coefficient_v<C11_1> == 1);
  static_assert(constant_matrix<C11_m1>);
  static_assert(constant_coefficient_v<C11_m1> == -1);
  static_assert(constant_coefficient_v<decltype(M11::Identity().conjugate())> == 1);
  static_assert(constant_coefficient_v<C11_2> == 2);
  static_assert(constant_coefficient_v<C11_m2> == -2);
  static_assert(constant_coefficient_v<C20_2> == 2);
  static_assert(constant_coefficient_v<C02_2> == 2);
  static_assert(constant_coefficient_v<B22_true> == true);
  static_assert(constant_coefficient_v<B22_false> == false);
  static_assert(constant_coefficient_v<decltype(c22_2.block<2,1>(0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.block(2, 1, 0, 0))> == 2);

  static_assert(constant_coefficient_v<decltype(c21_2 - c21_2)> == 0);
  static_assert(constant_coefficient_v<decltype(c21_2 + c21_m2)> == 0);
  static_assert(constant_coefficient_v<decltype(c21_2 - c21_m2)> == 4);
  static_assert(constant_coefficient_v<decltype(c21_2.array() * c21_m2.array())> == -4);
  static_assert(constant_coefficient_v<decltype(c21_2.array() / c21_m2.array())> == -1);
  static_assert(constant_coefficient_v<decltype(c21_2.array().min(c21_m2.array()))> == -2);
  static_assert(constant_coefficient_v<decltype(c21_2.array().max(c21_m2.array()))> == 2);
  static_assert(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, std::decay_t<decltype(c21_2.array())>, std::decay_t<decltype(c21_m2.array())>> {c21_2.array(), c21_m2.array()})> == 2);
  static_assert(constant_coefficient_v<decltype(c21_m2.array().pow(c21_3.array()))> == -8);
  static_assert(constant_coefficient_v<decltype(b22_true.array() and b22_true.array())> == true);
  static_assert(constant_coefficient_v<decltype(b22_true.array() and b22_false.array())> == false);
  static_assert(constant_coefficient_v<decltype(b22_false.array() or b22_true.array())> == true);
  static_assert(constant_coefficient_v<decltype(b22_false.array() or b22_false.array())> == false);
  static_assert(constant_coefficient_v<decltype(b22_false.array() xor b22_true.array())> == true);
  static_assert(constant_coefficient_v<decltype(b22_true.array() xor b22_true.array())> == false);

  static_assert(constant_coefficient_v<decltype(c11_2 * c11_2)> == 4);
  static_assert(constant_coefficient_v<decltype(c12_2 * c21_2)> == 8);
  static_assert(constant_coefficient_v<decltype(c11_1 * c11_2)> == 2);
  static_assert(constant_coefficient_v<decltype(c11_2 * c11_1)> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2 * c22_2)> == 8);
  static_assert(constant_coefficient_v<decltype(c22_2 * c22_m2)> == -8);
  static_assert(constant_coefficient_v<decltype(c22_2 * i22)> == 2);
  static_assert(constant_coefficient_v<decltype(i22 * c22_2)> == 2);

  static_assert(constant_coefficient_v<decltype(c22_2.diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(c11_2.asDiagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(z21.asDiagonal())> == 0);
  static_assert(not zero_matrix<decltype(c11_1)>);
  static_assert(not zero_matrix<decltype(c11_2)>);
  static_assert(not zero_matrix<decltype(c21_2)>);
  static_assert(not constant_matrix<decltype(c21_2.asDiagonal())>);
  static_assert(constant_coefficient_v<decltype(c22_2.array().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.replicate<5,5>())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(b22_true.select(c22_2, z22))> == 2);
  static_assert(constant_coefficient_v<decltype(b22_false.select(c22_2, z22))> == 0);
  static_assert(constant_coefficient_v<decltype(eigen_matrix_t<bool, 2, 2> {true, false, true, false}.select(c22_2, c22_2))> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.transpose())> == 2);
  static_assert(constant_coefficient_v<decltype(C22_2 {c22_2}.real())> == 2);
  static_assert(constant_coefficient_v<decltype(C22_2 {c22_2}.imag())> == 0);
  static_assert(constant_coefficient_v<decltype(c22_2.real())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.imag())> == 0);
  static_assert(constant_coefficient_v<decltype(c22_2.array().abs())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.array().abs())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.array().abs2())> == 4);
  static_assert(constant_coefficient_v<decltype(c22_2.array().sqrt())> == 1);
  static_assert(constant_coefficient_v<decltype(c22_1.array().inverse())> == 1);
  static_assert(constant_coefficient_v<decltype(c22_2.array().inverse())> == 0);
  static_assert(constant_coefficient_v<decltype(c22_m2.array().square())> == 4);
  static_assert(constant_coefficient_v<decltype(c22_m2.array().cube())> == -8);
  static_assert(constant_coefficient_v<decltype(not b22_true.array())> == false);
  static_assert(constant_coefficient_v<decltype(not b22_false.array())> == true);
  static_assert(constant_coefficient_v<decltype(not c22_1.array())> == false); // requires narrowing from 1 to true.
  static_assert(constant_coefficient_v<decltype(not z22.array())> == true); // requires narrowing from 0 to false.

  static_assert(constant_coefficient_v<Eigen::SelfAdjointView<C22_2, Eigen::Upper>> == 2);
  static_assert(constant_coefficient_v<Eigen::TriangularView<C22_2, Eigen::Lower>> == 2);
  static_assert(constant_coefficient_v<decltype(c21_2.segment<1>(0))> == 2);
  static_assert(constant_coefficient_v<decltype(c21_m2.segment(1, 0))> == -2);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.rowwise())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().squaredNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().norm())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().stableNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().hypotNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(c20_2_2.colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(c02_2_2.rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(c21_2.colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(c21_2.rowwise().sum())> == 2);
  static_assert(constant_coefficient_v<decltype(c20_1_2.colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(c01_2_2.rowwise().sum())> == 2);
  static_assert(not constant_matrix<decltype(c02_2_2.colwise().sum())>);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().mean())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(b22_true.colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(b22_true.colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(c20_2_2.colwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(c02_2_2.rowwise().count())> == 2);
  static_assert(not constant_matrix<decltype(c02_2_2.colwise().count())>);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().prod())> == 4);
  static_assert(constant_coefficient_v<decltype(c20_2_2.colwise().prod())> == 4);
  static_assert(constant_coefficient_v<decltype(c02_2_2.rowwise().prod())> == 4);
  static_assert(not constant_matrix<decltype(c02_2_2.colwise().prod())>);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().replicate<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().replicate(2))> == 2);

  static_assert(not zero_matrix<decltype(M00::Zero(2, 3))>); // because the constant is not known at compile time
  static_assert(zero_matrix<Z11>);
  static_assert(zero_matrix<Z22>);
  static_assert(zero_matrix<Z21>);
  static_assert(zero_matrix<Z23>);
  static_assert(zero_matrix<Z20>);
  static_assert(zero_matrix<Z02>);
  static_assert(zero_matrix<decltype(z22.block<2,1>(0, 0))>);
  static_assert(zero_matrix<decltype(z22.block(2, 1, 0, 0))>);
  static_assert(zero_matrix<decltype(c21_2 - c21_2)>);
  static_assert(zero_matrix<decltype(c21_2 + c21_m2)>);
  static_assert(zero_matrix<decltype(z11 * z11)>);
  static_assert(zero_matrix<decltype(z12 * z21)>);
  static_assert(zero_matrix<decltype(z11.asDiagonal())>);
  static_assert(zero_matrix<decltype(z21.asDiagonal())>);
  static_assert(zero_matrix<decltype(z21.array())>);
  static_assert(zero_matrix<decltype(z23.array())>);
  static_assert(zero_matrix<decltype(z23.array().matrix())>);
  static_assert(zero_matrix<decltype(z23.reverse())>);
  static_assert(constant_matrix<B22_true>);
  static_assert(zero_matrix<decltype(b22_true.select(z22, M22::Identity()))>);
  static_assert(not zero_matrix<decltype(b22_true.select(i22, z22))>);
  static_assert(zero_matrix<decltype(b22_false.select(i22, z22))>);
  static_assert(zero_matrix<decltype(eigen_matrix_t<bool, 2, 2> {true, false, true, false}.select(z22, z22))>);
  static_assert(zero_matrix<decltype((z23).transpose())>);
  static_assert(zero_matrix<Eigen::SelfAdjointView<Z22, Eigen::Upper>>);
  static_assert(zero_matrix<Eigen::TriangularView<Z22, Eigen::Lower>>);
  static_assert(zero_matrix<decltype(z21.segment<1>(0))>);
  static_assert(zero_matrix<decltype(z21.segment(1, 0))>);
  static_assert(zero_matrix<decltype(z22.colwise())>);
  static_assert(zero_matrix<decltype(z22.rowwise())>);
  static_assert(zero_matrix<decltype(z22.colwise().lpNorm<1>())>);
  static_assert(zero_matrix<decltype(z22.colwise().lpNorm<2>())>);
  static_assert(zero_matrix<decltype(z22.colwise().squaredNorm())>);
  static_assert(zero_matrix<decltype(z22.colwise().norm())>);
  static_assert(zero_matrix<decltype(z22.colwise().stableNorm())>);
  static_assert(zero_matrix<decltype(z22.colwise().hypotNorm())>);
  static_assert(zero_matrix<decltype(z22.colwise().sum())>);
  static_assert(zero_matrix<decltype(z22.colwise().mean())>);
  static_assert(zero_matrix<decltype(z22.colwise().minCoeff())>);
  static_assert(zero_matrix<decltype(z22.colwise().maxCoeff())>);
  static_assert(zero_matrix<decltype(z22.colwise().prod())>);
  static_assert(zero_matrix<decltype(z22.colwise().reverse())>);
  static_assert(zero_matrix<decltype(z22.colwise().replicate<2>())>);
  static_assert(zero_matrix<decltype(z22.colwise().replicate(2))>);

  static_assert(identity_matrix<C11_1>);
  static_assert(identity_matrix<decltype(-c11_m1)>);

  static_assert(identity_matrix<I22>);
  static_assert(identity_matrix<I20_2>);
  static_assert(identity_matrix<I02_2>);
  static_assert(identity_matrix<I00_22>);

  static_assert(identity_matrix<decltype(M33::Identity())>);
  static_assert(not identity_matrix<decltype(M30::Identity(3, 3))>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<decltype(M03::Identity(3, 3))>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<decltype(M00::Identity(3, 3))>); // We can't tell if it's a square matrix at compile time
  static_assert(identity_matrix<typename M33::IdentityReturnType>);
  static_assert(identity_matrix<decltype(M33::Identity() - (M33::Identity() - M33::Identity()))>);

  static_assert(identity_matrix<decltype(c11_1.asDiagonal())>);
  static_assert(identity_matrix<decltype(c21_1.asDiagonal())>);
  static_assert(identity_matrix<decltype(c21_1.asDiagonal().array())>);

  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(not identity_matrix<Eigen::SelfAdjointView<decltype(M30::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::SelfAdjointView<decltype(M03::Identity(3, 3)), Eigen::Upper>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::SelfAdjointView<decltype(M00::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time

  static_assert(identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M30::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M03::Identity(3, 3)), Eigen::Upper>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M00::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time

  static_assert(identity_matrix<Eigen::Replicate<I22, 1, 1>>);
  static_assert(identity_matrix<Eigen::Replicate<I20_2, 1, 1>>);
  static_assert(identity_matrix<Eigen::Replicate<I02_2, 1, 1>>);
  static_assert(identity_matrix<Eigen::Replicate<I00_22, 1, 1>>);

  static_assert(identity_matrix<Eigen::Reverse<I22, Eigen::BothDirections>>);
  static_assert(identity_matrix<Eigen::Reverse<I20_2, Eigen::BothDirections>>);
  static_assert(identity_matrix<Eigen::Reverse<I02_2, Eigen::BothDirections>>);
  static_assert(identity_matrix<Eigen::Reverse<I00_22, Eigen::BothDirections>>);

  static_assert(diagonal_matrix<I22>);
  static_assert(diagonal_matrix<I20_2>);
  static_assert(diagonal_matrix<I02_2>);
  static_assert(diagonal_matrix<I00_22>);
  static_assert(diagonal_matrix<D22_2>);
  static_assert(diagonal_matrix<D20_2_2>);
  static_assert(diagonal_matrix<D02_2_2>);
  static_assert(diagonal_matrix<D00_22_2>);

  static_assert(diagonal_matrix<decltype(c11_2.asDiagonal())>);
  static_assert(diagonal_matrix<decltype(c21_2.asDiagonal())>);
  static_assert(diagonal_matrix<decltype(c21_2.asDiagonal().array())>);

  static_assert(diagonal_matrix<Eigen::DiagonalMatrix<double, 3>>);
  static_assert(diagonal_matrix<Eigen::DiagonalWrapper<M31>>);
  static_assert(diagonal_matrix<typename M33::IdentityReturnType>);
  static_assert(diagonal_matrix<MatrixTraits<M22>::template DiagonalMatrixFrom<>>);

  static_assert(self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(self_adjoint_matrix<typename M33::ConstantReturnType>);
  static_assert(self_adjoint_matrix<typename M33::IdentityReturnType>);

  static_assert(upper_triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>>);
  static_assert(upper_triangular_matrix<typename M33::IdentityReturnType>);
  static_assert(upper_triangular_matrix<MatrixTraits<M22>::template TriangularMatrixFrom<TriangleType::upper>>);

  static_assert(lower_triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>>);
  static_assert(lower_triangular_matrix<typename M33::IdentityReturnType>);

  static_assert(lower_self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);

  static_assert(upper_self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(upper_self_adjoint_matrix<MatrixTraits<M22>::template SelfAdjointMatrixFrom<TriangleType::upper>>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>>);
  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>>);

  auto id2 = MatrixTraits<M22>::identity();

  static_assert(std::is_same_v<native_matrix_t<M33>, M33>);
  static_assert(std::is_same_v<native_matrix_t<M30>, M30>);
  static_assert(std::is_same_v<native_matrix_t<M03>, M03>);
  static_assert(std::is_same_v<native_matrix_t<M00>, M00>);

  static_assert(std::is_same_v<native_matrix_t<Eigen::SelfAdjointView<M33, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::SelfAdjointView<M30, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::SelfAdjointView<M03, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::SelfAdjointView<M00, Eigen::Lower>>, M00>);

  static_assert(std::is_same_v<native_matrix_t<Eigen::TriangularView<M33, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::TriangularView<M30, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::TriangularView<M03, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::TriangularView<M00, Eigen::Upper>>, M00>);

  static_assert(std::is_same_v<native_matrix_t<Eigen::DiagonalMatrix<double, 3>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>, M00>);

  static_assert(std::is_same_v<native_matrix_t<Eigen::DiagonalWrapper<M31>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::DiagonalWrapper<M30>>, M33>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::DiagonalWrapper<M01>>, M00>);
  static_assert(std::is_same_v<native_matrix_t<Eigen::DiagonalWrapper<M00>>, M00>);

  static_assert(self_contained<const M22>);
  static_assert(self_contained<typename M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::IdentityReturnType>);
  static_assert(self_contained<decltype(2 * id2 + id2)>);
  static_assert(not self_contained<decltype(2 * id2 + M22 {1, 2, 3, 4})>);
  static_assert(MatrixTraits<std::remove_const_t<decltype(2 * id2 + id2)>>::rows == 2);
  static_assert(self_contained<decltype(column<0>(2 * id2 + id2))>);
  static_assert(self_contained<decltype(column<0>(2 * id2 + M22 {1, 2, 3, 4}))>);
  static_assert(self_contained<decltype(row<0>(2 * id2 + id2))>);
  static_assert(self_contained<decltype(row<0>(2 * id2 + M22 {1, 2, 3, 4}))>);
  static_assert(self_contained<const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const M22, const M22>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const M22,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>>);

  static_assert(dynamic_rows<eigen_matrix_t<double, 0, 0>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, 0, 0>>);
  static_assert(dynamic_rows<eigen_matrix_t<double, 0, 1>>);
  static_assert(not dynamic_columns<eigen_matrix_t<double, 0, 1>>);
  static_assert(not dynamic_rows<eigen_matrix_t<double, 1, 0>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, 1, 0>>);

  static_assert(not dynamic_shape<native_matrix_t<SelfAdjointMatrix<M02>>>);
  static_assert(not dynamic_shape<native_matrix_t<SelfAdjointMatrix<M20>>>);

  static_assert(element_gettable<M32, 2>);
  static_assert(element_gettable<const M32, 2>);
  static_assert(element_gettable<M31, 2>);
  static_assert(element_gettable<M13, 2>);
  static_assert(element_gettable<M30, 2>);
  static_assert(element_gettable<M02, 2>);
  static_assert(element_gettable<M01, 2>);
  static_assert(element_gettable<M10, 2>);
  static_assert(element_gettable<M00, 2>);

  static_assert(not element_gettable<M32, 1>);
  static_assert(element_gettable<M31, 1>);
  static_assert(element_gettable<M13, 1>);
  static_assert(not element_gettable<M02, 1>);
  static_assert(element_gettable<M01, 1>);
  static_assert(element_gettable<M10, 1>);
  static_assert(not element_gettable<M00, 1>);

  static_assert(element_settable<M32, 2>);
  static_assert(not element_settable<const M32, 2>);
  static_assert(element_settable<M31, 2>);
  static_assert(element_settable<M13, 2>);
  static_assert(element_settable<M30, 2>);
  static_assert(element_settable<M02, 2>);
  static_assert(element_settable<M01, 2>);
  static_assert(element_settable<M10, 2>);
  static_assert(element_settable<M00, 2>);

  static_assert(not element_settable<M32, 1>);
  static_assert(element_settable<M31, 1>);
  static_assert(element_settable<M13, 1>);
  static_assert(not element_settable<const M31, 1>);
  static_assert(not element_settable<M02, 1>);
  static_assert(element_settable<M01, 1>);
  static_assert(element_settable<M10, 1>);
  static_assert(not element_settable<M00, 1>);

  static_assert(writable<MatrixTraits<M22>::template NativeMatrixFrom<>>);
  static_assert(writable<Eigen::DiagonalMatrix<double, 3>>);
  static_assert(writable<Eigen::DiagonalWrapper<M31>>);

  static_assert(std::is_same_v<MatrixTraits<Eigen::SelfAdjointView<M22, Eigen::Lower>>::NestedMatrix, M22&>);
  static_assert(std::is_same_v<MatrixTraits<Eigen::TriangularView<M22, Eigen::Upper>>::NestedMatrix, M22&>);
  static_assert(std::is_same_v<MatrixTraits<Eigen::DiagonalMatrix<double, 2>>::NestedMatrix, M21>);
  static_assert(std::is_same_v<MatrixTraits<Eigen::DiagonalWrapper<M21>>::NestedMatrix, M21&>);

  M22 m22; m22 << 1, 2, 3, 4;
  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  M03 m03_2 {2,3}; m03_2 << 1, 2, 3, 4, 5, 6;
  M32 m32; m32 << 1, 4, 2, 5, 3, 6;
  CM22 cm22; cm22 << cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1};

  EXPECT_TRUE(is_near(MatrixTraits<M22>::make(m22), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(m23), m23));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(m22), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(m32), m32));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(m22), m22));
  EXPECT_TRUE(is_near(MatrixTraits<CM22>::make(cm22), cm22));
  static_assert(MatrixTraits<decltype(MatrixTraits<M20>::make(m23))>::columns == 3);
  static_assert(MatrixTraits<decltype(MatrixTraits<M03>::make(m03_2))>::rows == 0);

  EXPECT_TRUE(is_near(MatrixTraits<M22>::make(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(MatrixTraits<M00>::make(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(MatrixTraits<CM22>::make(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(MatrixTraits<decltype(MatrixTraits<M20>::make(1, 2))>::columns == 1);
  static_assert(MatrixTraits<decltype(MatrixTraits<M02>::make(1, 2))>::rows == 1);
  static_assert(MatrixTraits<decltype(MatrixTraits<M00>::make(1, 2))>::rows == 2);

  EXPECT_TRUE(is_near(MatrixTraits<M23>::zero(), M23::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::zero(3), M23::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<M03>::zero(2), M23::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<M00>::zero(2,3), M23::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<M23>::zero(3,2), M32::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::zero(3,2), M32::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<M03>::zero(3,2), M32::Zero()));

  EXPECT_TRUE(is_near(MatrixTraits<M33>::identity(), M33::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<M30>::identity(), M33::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<M03>::identity(), M33::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<M00>::identity(3), M33::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<M33>::identity(4), M44::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<M30>::identity(4), M44::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<M03>::identity(4), M44::Identity()));
}


TEST(eigen3, Eigen_Matrix_overloads)
{
  M21 m21; m21 << 1, 4;
  M20 m20_1 {2,1}; m20_1 << 1, 4;
  M01 m01_2 {2,1}; m01_2 << 1, 4;
  M02 m02_1 {1,2}; m02_1 << 1, 4;
  M00 m00_21 {2,1}; m00_21 << 1, 4;

  M22 m22; m22 << 1, 2, 3, 4;
  M20 m20_2 {2,2}; m20_2 << 1, 2, 3, 4;
  M02 m02_2 {2,2}; m02_2 << 1, 2, 3, 4;
  M00 m00_22 {2,2}; m00_22 << 1, 2, 3, 4;

  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  M03 m03_2 {2,3}; m03_2 << 1, 2, 3, 4, 5, 6;
  M20 m20_3 {2,3}; m20_3 << 1, 2, 3, 4, 5, 6;
  M00 m00_23 {2,3}; m00_23 << 1, 2, 3, 4, 5, 6;

  M32 m32; m32 << 1, 4, 2, 5, 3, 6;
  M02 m02_3 {3,2}; m02_3 << 1, 4, 2, 5, 3, 6;
  M30 m30_2 {3,2}; m30_2 << 1, 4, 3, 5, 3, 6;
  M30 m00_32 {3,2}; m00_32 << 1, 4, 3, 5, 3, 6;

  M11 m11; m11 << 3;
  M10 m10_1 {1,1}; m10_1 << 3;
  M01 m01_1 {1,1}; m01_1 << 3;
  M00 m00_11 {1,1}; m00_11 << 3;

  CM22 cm22; cm22 << cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1};
  CM23 cm23; cm23 << cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1};
  CM32 cm32; cm32 << cdouble {1,6}, cdouble {4,3}, cdouble {2,5}, cdouble {5,2}, cdouble {3,4}, cdouble {6,1};
  CM32 cm32conj; cm32conj << cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1};

  auto m22_93310 = make_native_matrix<M22>(9, 3, 3, 10);
  auto m20_93310 = M20 {m22_93310};
  auto m02_93310 = M02 {m22_93310};
  auto m00_93310 = M00 {m22_93310};

  auto cm22_93310 = make_native_matrix<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);
  auto cm20_93310 = CM20 {cm22_93310};
  auto cm02_93310 = CM02 {cm22_93310};
  auto cm00_93310 = CM00 {cm22_93310};

  auto m22_3013 = make_native_matrix<M22>(3, 0, 1, 3);
  auto m20_3013 = M20 {m22_3013};
  auto m02_3013 = M02 {m22_3013};
  auto m00_3013 = M00 {m22_3013};

  auto cm22_3013 = make_native_matrix<CM22>(cdouble(3,1), 0, cdouble(1,1), 3);
  auto cm20_3013 = CM20 {cm22_3013};
  auto cm02_3013 = CM02 {cm22_3013};
  auto cm00_3013 = CM00 {cm22_3013};

  auto m22_3103 = make_native_matrix<M22>(3, 1, 0, 3);
  auto m20_3103 = M20 {m22_3103};
  auto m02_3103 = M02 {m22_3103};
  auto m00_3103 = M00 {m22_3103};

  auto cm22_3103 = make_native_matrix<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);
  auto cm20_3103 = CM20 {cm22_3103};
  auto cm02_3103 = CM02 {cm22_3103};
  auto cm00_3103 = CM00 {cm22_3103};

  EXPECT_TRUE(is_near(nested_matrix(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}), m22_93310));
  EXPECT_TRUE(is_near(nested_matrix(Eigen::TriangularView<M22, Eigen::Upper> {m22_3103}), m22_3103));
  EXPECT_TRUE(is_near(nested_matrix(Eigen::DiagonalMatrix<double, 2> {m21}), m21));
  EXPECT_TRUE(is_near(nested_matrix(Eigen::DiagonalWrapper<M21> {m21}), m21));

  EXPECT_NEAR(m22(0, 0), 1, 1e-6);
  EXPECT_NEAR(m22(0, 1), 2, 1e-6);
  auto d1 = make_native_matrix<double, 3, 1>(1, 2, 3);
  EXPECT_NEAR(d1(1), 2, 1e-6);
  d1(0) = 5;
  d1(2, 0) = 7;
  EXPECT_TRUE(is_near(d1, make_native_matrix<double, 3, 1>(5, 2, 7)));

  EXPECT_TRUE(is_near(make_native_matrix<M22>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<M20>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_native_matrix<M20>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<M20>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<M02>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_native_matrix<M02>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<M02>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<M00>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(MatrixTraits<decltype(make_native_matrix<M20>(1, 2))>::columns == 1);
  static_assert(MatrixTraits<decltype(make_native_matrix<M02>(1, 2))>::rows == 1);
  static_assert(MatrixTraits<decltype(make_native_matrix<M00>(1, 2))>::rows == 2);

  EXPECT_TRUE(is_near(make_native_matrix<double, 2, 2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<double, 2, 0>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_native_matrix<double, 2, 0>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<double, 2, 0>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<double, 0, 2>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_native_matrix<double, 0, 2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<double, 0, 2>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<double, 0, 0>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<cdouble, 2, 2>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(MatrixTraits<decltype(make_native_matrix<double, 2, 0>(1, 2))>::columns == 1);
  static_assert(MatrixTraits<decltype(make_native_matrix<double, 0, 2>(1, 2))>::rows == 1);
  static_assert(MatrixTraits<decltype(make_native_matrix<double, 0, 0>(1, 2))>::rows == 2);

  EXPECT_TRUE(is_near(make_native_matrix<2, 2>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<2, 0>(1., 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_native_matrix<2, 0>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<2, 0>(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<0, 2>(1., 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_native_matrix<0, 2>(1., 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_native_matrix<0, 2>(1., 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<0, 0>(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix<2, 2>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(MatrixTraits<decltype(make_native_matrix<2, 0>(1., 2))>::columns == 1);
  static_assert(MatrixTraits<decltype(make_native_matrix<0, 2>(1., 2))>::rows == 1);
  static_assert(MatrixTraits<decltype(make_native_matrix<0, 0>(1., 2))>::rows == 2);

  EXPECT_TRUE(is_near(make_native_matrix(1., 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_native_matrix(1., 2, 3, 4), (eigen_matrix_t<double, 4, 1> {} << 1, 2, 3, 4).finished()));

  EXPECT_TRUE(is_near(make_native_matrix(m22), m22));

  EXPECT_TRUE(is_near(make_native_matrix<double, 1, 1>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_native_matrix<double, 1, 0>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_native_matrix<double, 0, 1>(4), eigen_matrix_t<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(make_native_matrix<double, 0, 0>(4), eigen_matrix_t<double, 1, 1>(4)));

  EXPECT_TRUE(is_near(make_native_matrix(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(make_native_matrix(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(make_native_matrix(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(make_native_matrix(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  EXPECT_TRUE(is_near(make_native_matrix(m22_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_native_matrix(m20_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(make_native_matrix(m02_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(make_native_matrix(m00_3103.template triangularView<Eigen::Upper>()), m22_3103));

  auto m22_1004 = make_native_matrix<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalMatrix<double, 2> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m00_21}), m22_1004));

  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalWrapper<M20> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalWrapper<M01> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(make_native_matrix(Eigen::DiagonalWrapper<M00> {m00_21}), m22_1004));

  EXPECT_EQ(row_count(m23), 2);
  EXPECT_EQ(row_count(m20_3), 2);
  EXPECT_EQ(row_count(m03_2), 2);
  EXPECT_EQ(row_count(m00_23), 2);

  EXPECT_EQ(column_count(m23), 3);
  EXPECT_EQ(column_count(m20_3), 3);
  EXPECT_EQ(column_count(m03_2), 3);
  EXPECT_EQ(column_count(m00_23), 3);

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  EXPECT_TRUE(is_near(to_diagonal(m11), m11));
  EXPECT_TRUE(is_near(to_diagonal(m10_1), m11));
  EXPECT_TRUE(is_near(to_diagonal(m01_1), m11));
  EXPECT_TRUE(is_near(to_diagonal(m00_11), m11));
  EXPECT_TRUE(is_near(to_diagonal(m21), m22_1004));
  EXPECT_TRUE(is_near(to_diagonal(m20_1), m22_1004)); static_assert(not dynamic_shape<decltype(to_diagonal(m20_1))>);
  EXPECT_TRUE(is_near(to_diagonal(m01_2), m22_1004)); static_assert(dynamic_shape<decltype(to_diagonal(m01_2))>);
  EXPECT_TRUE(is_near(to_diagonal(m00_21), m22_1004)); static_assert(dynamic_shape<decltype(to_diagonal(m00_21))>);
  static_assert(zero_matrix<decltype(to_diagonal(z21))>);
  static_assert(zero_matrix<decltype(to_diagonal(z20_1))>); static_assert(not dynamic_shape<decltype(to_diagonal(z20_1))>);
  static_assert(zero_matrix<decltype(to_diagonal(z01_2))>); static_assert(dynamic_shape<decltype(to_diagonal(z01_2))>);
  static_assert(zero_matrix<decltype(to_diagonal(z00_21))>); static_assert(dynamic_shape<decltype(to_diagonal(z00_21))>);

  EXPECT_TRUE(is_near(diagonal_of(m11), m11));
  EXPECT_TRUE(is_near(diagonal_of(m10_1), m11)); static_assert(dynamic_shape<decltype(diagonal_of(m10_1))>);
  EXPECT_TRUE(is_near(diagonal_of(m01_1), m11)); static_assert(dynamic_shape<decltype(diagonal_of(m01_1))>);
  EXPECT_TRUE(is_near(diagonal_of(m00_11), m11)); static_assert(dynamic_shape<decltype(diagonal_of(m00_11))>);

  EXPECT_TRUE(is_near(diagonal_of(m22), m21));
  EXPECT_TRUE(is_near(diagonal_of(m20_2), m21)); static_assert(dynamic_shape<decltype(diagonal_of(m20_2))>);
  EXPECT_TRUE(is_near(diagonal_of(m02_2), m21)); static_assert(dynamic_shape<decltype(diagonal_of(m02_2))>);
  EXPECT_TRUE(is_near(diagonal_of(m00_22), m21)); static_assert(dynamic_shape<decltype(diagonal_of(m00_22))>);

  EXPECT_TRUE(is_near(diagonal_of(M22 {m22}), m21));
  EXPECT_TRUE(is_near(diagonal_of(M20 {m20_2}), m21)); static_assert(not dynamic_shape<decltype(diagonal_of(M20 {m20_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(M02 {m02_2}), m21)); static_assert(not dynamic_shape<decltype(diagonal_of(M02 {m02_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(M00 {m00_22}), m21)); static_assert(dynamic_shape<decltype(diagonal_of(M00 {m00_22}))>);

  M21 c21; c21 << 1, 1;

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity()), c21)); static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2, 2)), c21)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2, 2)))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M02::Identity(2, 2)), c21)); static_assert(constant_coefficient_v<decltype(diagonal_of(M02::Identity(2, 2)))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M00::Identity(2, 2)), c21)); static_assert(constant_coefficient_v<decltype(diagonal_of(M00::Identity(2, 2)))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(z22), z21));
  EXPECT_TRUE(is_near(diagonal_of(z20_2), z21)); static_assert(zero_matrix<decltype(diagonal_of(z20_2))>); static_assert(not dynamic_shape<decltype(diagonal_of(z20_2))>);
  EXPECT_TRUE(is_near(diagonal_of(z02_2), z21)); static_assert(zero_matrix<decltype(diagonal_of(z02_2))>); static_assert(not dynamic_shape<decltype(diagonal_of(z02_2))>);
  EXPECT_TRUE(is_near(diagonal_of(z00_22), z21)); static_assert(zero_matrix<decltype(diagonal_of(z00_22))>); static_assert(dynamic_shape<decltype(diagonal_of(z00_22))>);

  EXPECT_TRUE(is_near(diagonal_of(c22_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c22_2))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c20_2_2))> == 2); static_assert(not dynamic_shape<decltype(diagonal_of(c20_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c02_2_2))> == 2); static_assert(not dynamic_shape<decltype(diagonal_of(c02_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c00_22_2))> == 2); static_assert(dynamic_shape<decltype(diagonal_of(c00_22_2))>);

  EXPECT_TRUE(is_near(diagonal_of(i22), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(i22))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(i20_2), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(i20_2))> == 1); static_assert(not dynamic_shape<decltype(diagonal_of(i20_2))>);
  EXPECT_TRUE(is_near(diagonal_of(i02_2), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(i02_2))> == 1); static_assert(dynamic_shape<decltype(diagonal_of(i02_2))>);
  EXPECT_TRUE(is_near(diagonal_of(i00_22), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(i00_22))> == 1); static_assert(dynamic_shape<decltype(diagonal_of(i00_22))>);

  EXPECT_TRUE(is_near(diagonal_of(d22_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d22_2))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(d20_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d20_2_2))> == 2); static_assert(dynamic_shape<decltype(diagonal_of(d20_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(d02_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d02_2_2))> == 2); static_assert(dynamic_shape<decltype(diagonal_of(d02_2_2))>);
  EXPECT_TRUE(is_near(diagonal_of(d00_22_2), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(d00_22_2))> == 2); static_assert(dynamic_shape<decltype(diagonal_of(d00_22_2))>);

  auto m21_910 = make_native_matrix<M21>(9, 10);

  auto m21_33 = make_native_matrix<M21>(3, 3);
  auto m20_33 = make_native_matrix<M20>(3, 3);
  auto m01_33 = make_native_matrix<M01>(3, 3);
  auto m00_33 = make_native_matrix<M00>(3, 3);

  EXPECT_TRUE(is_near(diagonal_of(m22_93310.template selfadjointView<Eigen::Upper>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m20_93310.template selfadjointView<Eigen::Lower>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m02_93310.template selfadjointView<Eigen::Upper>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m00_93310.template selfadjointView<Eigen::Lower>()), m21_910));

  EXPECT_TRUE(is_near(diagonal_of(z22.template selfadjointView<Eigen::Upper>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z22.template selfadjointView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z20_2.template selfadjointView<Eigen::Lower>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z20_2.template selfadjointView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z02_2.template selfadjointView<Eigen::Upper>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z02_2.template selfadjointView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z00_22.template selfadjointView<Eigen::Lower>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z00_22.template selfadjointView<Eigen::Lower>()))>);

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template selfadjointView<Eigen::Upper>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c22_2.template selfadjointView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2.template selfadjointView<Eigen::Lower>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c20_2_2.template selfadjointView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2.template selfadjointView<Eigen::Upper>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c02_2_2.template selfadjointView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2.template selfadjointView<Eigen::Lower>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c00_22_2.template selfadjointView<Eigen::Lower>()))> == 2);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template selfadjointView<Eigen::Upper>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity().template selfadjointView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Upper>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template selfadjointView<Eigen::Lower>()))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(m22_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m20_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m02_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m00_3103.template triangularView<Eigen::Upper>()), m21_33));

  EXPECT_TRUE(is_near(diagonal_of(z22.template triangularView<Eigen::Upper>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z22.template triangularView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z20_2.template triangularView<Eigen::Lower>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z20_2.template triangularView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z02_2.template triangularView<Eigen::Upper>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z02_2.template triangularView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z00_22.template triangularView<Eigen::Lower>()), z21)); static_assert(zero_matrix<decltype(diagonal_of(z00_22.template triangularView<Eigen::Lower>()))>);

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template triangularView<Eigen::Upper>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c22_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()))> == 2);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Upper>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()), c21_1)); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);

  auto dm2 = Eigen::DiagonalMatrix<double, 2> {m21};
  auto dm0_2 = Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m01_2};

  EXPECT_TRUE(is_near(diagonal_of(dm2), m21));
  EXPECT_TRUE(is_near(diagonal_of(dm0_2), m01_2));

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalMatrix<double, 2> {dm2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {dm0_2}), m21));

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<M21> {m21}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<M20> {m20_1}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<M01> {m01_2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<M00> {m00_21}), m21));

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<Z21> {z21}), z21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper<Z21> {z21}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<Z20> {z20_1}), z21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper<Z20> {z20_1}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<Z01> {z01_2}), z21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper<Z01> {z01_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<Z00> {z00_21}), z21)); static_assert(zero_matrix<decltype(diagonal_of(Eigen::DiagonalWrapper<Z00> {z00_21}))>);

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<C21_2> {c21_2}), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(Eigen::DiagonalWrapper<C21_2> {c21_2}))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<C20_2> {c20_1_2}), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(Eigen::DiagonalWrapper<C20_2> {c20_1_2}))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<C01_2> {c01_2_2}), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(Eigen::DiagonalWrapper<C01_2> {c01_2_2}))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper<C00_2> {c00_21_2}), c21_2)); static_assert(constant_coefficient_v<decltype(diagonal_of(Eigen::DiagonalWrapper<C00_2> {c00_21_2}))> == 2);

  EXPECT_TRUE(is_near(transpose(m23), m32));
  EXPECT_TRUE(is_near(transpose(m02_3), m23));
  EXPECT_TRUE(is_near(transpose(m20_3), m32));
  EXPECT_TRUE(is_near(transpose(m00_23), m32));

  EXPECT_TRUE(is_near(transpose(cm23), cm32));

  EXPECT_TRUE(is_near(transpose(m22_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m20_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m02_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m00_93310.template selfadjointView<Eigen::Upper>()), m22_93310));

  EXPECT_TRUE(is_near(transpose(m22_3103.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(transpose(m20_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(transpose(m02_3103.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(transpose(m00_3103.template triangularView<Eigen::Upper>()), m22_3013));

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M20> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M01> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M00> {m00_21}), m22_1004));

  EXPECT_TRUE(is_near(transpose(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310));

  EXPECT_TRUE(is_near(transpose(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013));

  EXPECT_TRUE(is_near(transpose(z21), z12)); static_assert(zero_matrix<decltype(transpose(z21))>);
  EXPECT_TRUE(is_near(transpose(z20_1), z12)); static_assert(zero_matrix<decltype(transpose(z20_1))>);
  EXPECT_TRUE(is_near(transpose(z01_2), z12)); static_assert(zero_matrix<decltype(transpose(z01_2))>);
  EXPECT_TRUE(is_near(transpose(z00_21), z12)); static_assert(zero_matrix<decltype(transpose(z00_21))>);

  EXPECT_TRUE(is_near(transpose(c21_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c21_2))> == 2);
  EXPECT_TRUE(is_near(transpose(c20_1_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c20_1_2))> == 2);
  EXPECT_TRUE(is_near(transpose(c01_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c01_2_2))> == 2);
  EXPECT_TRUE(is_near(transpose(c00_21_2), c12_2)); static_assert(constant_coefficient_v<decltype(transpose(c00_21_2))> == 2);

  EXPECT_TRUE(is_near(adjoint(m23), m32));
  EXPECT_TRUE(is_near(adjoint(m02_3), m23));
  EXPECT_TRUE(is_near(adjoint(m20_3), m32));
  EXPECT_TRUE(is_near(adjoint(m00_23), m32));

  EXPECT_TRUE(is_near(adjoint(cm23), cm32conj));

  EXPECT_TRUE(is_near(adjoint(m22_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m02_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  EXPECT_TRUE(is_near(adjoint(m22_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m20_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m02_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m00_3103.template triangularView<Eigen::Upper>()), m22_3013));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M20> {m20_1}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M01> {m01_2}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M00> {m00_21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310));

  EXPECT_TRUE(is_near(adjoint(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013));

  EXPECT_TRUE(is_near(adjoint(z21), z12)); static_assert(zero_matrix<decltype(adjoint(z21))>);
  EXPECT_TRUE(is_near(adjoint(z20_1), z12)); static_assert(zero_matrix<decltype(adjoint(z20_1))>);
  EXPECT_TRUE(is_near(adjoint(z01_2), z12)); static_assert(zero_matrix<decltype(adjoint(z01_2))>);
  EXPECT_TRUE(is_near(adjoint(z00_21), z12)); static_assert(zero_matrix<decltype(adjoint(z00_21))>);

  EXPECT_TRUE(is_near(adjoint(c21_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c21_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(c20_1_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c20_1_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(c01_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c01_2_2))> == 2);
  EXPECT_TRUE(is_near(adjoint(c00_21_2), c12_2)); static_assert(constant_coefficient_v<decltype(adjoint(c00_21_2))> == 2);

  EXPECT_NEAR(determinant(m22), -2, 1e-6);
  EXPECT_NEAR(determinant(eigen_matrix_t<double, 1, 1> {2}), 2, 1e-6);
  EXPECT_NEAR(determinant(m02_2), -2, 1e-6);
  EXPECT_NEAR(determinant(m20_2), -2, 1e-6);
  EXPECT_NEAR(determinant(m00_22), -2, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22)), 0, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22)), 4, 1e-6);

  EXPECT_NEAR(determinant(m22_93310.template selfadjointView<Eigen::Lower>()), 81, 1e-6);

  EXPECT_NEAR(determinant(m22_3103.template triangularView<Eigen::Upper>()), 9, 1e-6);

  EXPECT_NEAR(determinant(Eigen::DiagonalMatrix<double, 2> {m21}), 4, 1e-6);

  EXPECT_NEAR(determinant(Eigen::DiagonalWrapper<M21> {m21}), 4, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Lower>())), 82, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22_3103.template triangularView<Eigen::Upper>())), 9, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3103.template triangularView<Eigen::Upper>())), 3, 1e-6);

  EXPECT_NEAR(determinant(z22), 0, 1e-6);
  EXPECT_NEAR(determinant(z20_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z02_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z00_22), 0, 1e-6);

  EXPECT_NEAR(determinant(c22_2), 0, 1e-6);
  EXPECT_NEAR(determinant(c20_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(c02_2_2), 0, 1e-6);
  EXPECT_NEAR(determinant(c00_22_2), 0, 1e-6);

  EXPECT_NEAR(determinant(M22::Identity()), 1, 1e-6);

  EXPECT_NEAR(trace(m22), 5, 1e-6);
  EXPECT_NEAR(trace(eigen_matrix_t<double, 1, 1> {3}), 3, 1e-6);
  EXPECT_NEAR(trace(m02_2), 5, 1e-6);
  EXPECT_NEAR(trace(m20_2), 5, 1e-6);
  EXPECT_NEAR(trace(m00_22), 5, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22)), 5, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22)), 5, 1e-6);

  EXPECT_NEAR(trace(m22_93310.template selfadjointView<Eigen::Lower>()), 19, 1e-6);

  EXPECT_NEAR(trace(m22_3103.template triangularView<Eigen::Upper>()), 6, 1e-6);

  EXPECT_NEAR(trace(Eigen::DiagonalMatrix<double, 2> {m21}), 5, 1e-6);

  EXPECT_NEAR(trace(Eigen::DiagonalWrapper<M21> {m21}), 5, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22_3103.template triangularView<Eigen::Upper>())), 6, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3103.template triangularView<Eigen::Upper>())), 1, 1e-6);

  EXPECT_NEAR(trace(z22), 0, 1e-6);
  EXPECT_NEAR(trace(z20_2), 0, 1e-6);
  EXPECT_NEAR(trace(z02_2), 0, 1e-6);
  EXPECT_NEAR(trace(z00_22), 0, 1e-6);

  EXPECT_NEAR(trace(c22_2), 4, 1e-6);
  EXPECT_NEAR(trace(c20_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(c02_2_2), 4, 1e-6);
  EXPECT_NEAR(trace(c00_22_2), 4, 1e-6);

  EXPECT_NEAR(trace(M22::Identity()), 2, 1e-6);

  auto m11_3 = m11;
  auto m10_1_3 = m10_1;
  auto m01_1_3 = m01_1;
  auto m00_11_3 = m00_11;

  auto m11_2 = make_native_matrix<M11>(2);
  auto m10_1_2 = M10 {m11_2};
  auto m01_1_2 = M01 {m11_2};
  auto m00_11_2 = M00 {m11_2};

  auto m11_5 = make_native_matrix<M11>(5);

  rank_update(m11_3, m11_2, 4); EXPECT_TRUE(is_near(m11_3, m11_5));
  rank_update(m10_1_3, m11_2, 4); EXPECT_TRUE(is_near(m10_1_3, m11_5));
  rank_update(m01_1_3, m11_2, 4); EXPECT_TRUE(is_near(m01_1_3, m11_5));
  rank_update(m00_11_3, m11_2, 4); EXPECT_TRUE(is_near(m00_11_3, m11_5));

  EXPECT_TRUE(is_near(rank_update(M11 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m10_1}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m01_1}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m00_11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M11 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m10_1}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m01_1}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m00_11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M11 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m10_1}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m01_1}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m00_11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M11 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m10_1}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m01_1}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m00_11}, M00 {m11_2}, 4), m11_5));
  static_assert(one_by_one_matrix<decltype(rank_update(M00 {m00_11}, M00 {m11_2}, 4))>);

  auto m22_5005 = make_native_matrix<M22>(5, 0, 0, 5);

  EXPECT_TRUE(is_near(rank_update(d22_3, d22_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, d22_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, d22_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, d22_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d22_3, d20_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, d20_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, d20_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, d20_2_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d22_3, d02_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, d02_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, d02_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, d02_2_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d22_3, d00_22_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, d00_22_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, d00_22_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, d00_22_2, 4), m22_5005));

  auto m22_2012 = make_native_matrix<M22>(2, 0, 1, 2);
  auto m20_2012 = M20 {m22_2012};
  auto m02_2012 = M02 {m22_2012};
  auto m00_2012 = M00 {m22_2012};

  auto m22_2102 = make_native_matrix<M22>(2, 1, 0, 2);
  auto m20_2102 = M20 {m22_2102};
  auto m02_2102 = M02 {m22_2102};
  auto m00_2102 = M00 {m22_2102};

  auto m22_25111130 = make_native_matrix<M22>(25, 11, 11, 30);
  auto m22_29111126 = make_native_matrix<M22>(29, 11, 11, 26);

  auto ru_93310_2102_4_rvalue = rank_update(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m22_2102, 4);
  EXPECT_TRUE(is_near(ru_93310_2102_4_rvalue, m22_29111126));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2102_4_rvalue)>);
  static_assert(upper_self_adjoint_matrix<decltype(ru_93310_2102_4_rvalue)>);
  static_assert(std::is_lvalue_reference_v<MatrixTraits<decltype(ru_93310_2102_4_rvalue)>::NestedMatrix>);

  auto sa_93310_2012_4 = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310};

  auto ru_93310_2012_4_lvalue = rank_update(sa_93310_2012_4, m22_2012, 4);
  EXPECT_TRUE(is_near(ru_93310_2012_4_lvalue, m22_25111130));
  static_assert(Eigen3::eigen_SelfAdjointView<decltype(ru_93310_2012_4_lvalue)>);
  static_assert(lower_self_adjoint_matrix<decltype(ru_93310_2012_4_lvalue)>);
  static_assert(std::is_lvalue_reference_v<MatrixTraits<decltype(ru_93310_2012_4_lvalue)>::NestedMatrix>);

  auto ru_93310_2012_4_const_lvalue = rank_update(std::as_const(sa_93310_2012_4), m22_2012, 4);
  EXPECT_TRUE(is_near(ru_93310_2012_4_const_lvalue, m22_25111130));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2012_4_const_lvalue)>);
  static_assert(not std::is_lvalue_reference_v<MatrixTraits<decltype(ru_93310_2012_4_const_lvalue)>::NestedMatrix>);

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}, m22_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Upper> {m20_93310}, m22_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Lower> {m02_93310}, m22_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Upper> {m00_93310}, m22_2102, 4), m22_29111126));

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m20_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Lower> {m20_93310}, m20_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Upper> {m02_93310}, m20_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Lower> {m00_93310}, m20_2012, 4), m22_25111130));

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}, m02_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Upper> {m20_93310}, m02_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Lower> {m02_93310}, m02_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Upper> {m00_93310}, m02_2102, 4), m22_29111126));

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m00_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Lower> {m20_93310}, m00_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Upper> {m02_93310}, m00_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Lower> {m00_93310}, m00_2012, 4), m22_25111130));


  const auto m22_50225 = make_native_matrix<M22>(5., 0, 2.2, std::sqrt(25.16));
  const auto m22_52025 = make_native_matrix<M22>(5., 2.2, 0, std::sqrt(25.16));

  auto m22_3013_rvalue {m22_3013};
  auto ru_3103_2102_4_rvalue = rank_update(Eigen::TriangularView<M22, Eigen::Upper> {m22_3013_rvalue}, m22_2102, 4);
  EXPECT_TRUE(is_near(ru_3103_2102_4_rvalue, m22_52025));
  static_assert(eigen_triangular_expr<decltype(ru_3103_2102_4_rvalue)>);
  static_assert(upper_triangular_matrix<decltype(ru_3103_2102_4_rvalue)>);
  static_assert(std::is_lvalue_reference_v<MatrixTraits<decltype(ru_3103_2102_4_rvalue)>::NestedMatrix>);

  auto m22_3013_lvalue {m22_3013};
  auto t_3013_2012_4_lvalue = Eigen::TriangularView<M22, Eigen::Lower> {m22_3013_lvalue};
  auto ru_3013_2012_4_lvalue = rank_update(t_3013_2012_4_lvalue, m22_2012, 4);
  EXPECT_TRUE(is_near(ru_3013_2012_4_lvalue, m22_50225));
  EXPECT_TRUE(is_near(t_3013_2012_4_lvalue, m22_50225));
  static_assert(triangular_matrix<decltype(ru_3013_2012_4_lvalue)>);
  static_assert(Eigen3::eigen_TriangularView<decltype(ru_3013_2012_4_lvalue)>);

  auto m22_3013_const_lvalue {m22_3013};
  auto t_3013_2012_4_const_lvalue = Eigen::TriangularView<M22, Eigen::Lower> {m22_3013_const_lvalue};
  auto ru_3013_2012_4_const_lvalue = rank_update(std::as_const(t_3013_2012_4_const_lvalue), m22_2012, 4);
  EXPECT_TRUE(is_near(ru_3013_2012_4_const_lvalue, m22_50225));
  static_assert(eigen_triangular_expr<decltype(ru_3013_2012_4_const_lvalue)>);
  static_assert(lower_triangular_matrix<decltype(ru_3013_2012_4_const_lvalue)>);
  static_assert(not std::is_lvalue_reference_v<MatrixTraits<decltype(ru_3013_2012_4_const_lvalue)>::NestedMatrix>);

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Lower> {m22_3013}, m22_2012, 4), m22_50225));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Upper> {m20_3103}, m22_2102, 4), m22_52025));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Lower> {m02_3013}, m22_2012, 4), m22_50225));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Upper> {m00_3103}, m22_2102, 4), m22_52025));

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Upper> {m22_3103}, m20_2102, 4), m22_52025));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Lower> {m20_3013}, m20_2012, 4), m22_50225));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Upper> {m02_3103}, m20_2102, 4), m22_52025));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Lower> {m00_3013}, m20_2012, 4), m22_50225));

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Lower> {m22_3013}, m02_2012, 4), m22_50225));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Upper> {m20_3103}, m02_2102, 4), m22_52025));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Lower> {m02_3013}, m02_2012, 4), m22_50225));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Upper> {m00_3103}, m02_2102, 4), m22_52025));

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Upper> {m22_3103}, m00_2102, 4), m22_52025));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Lower> {m20_3013}, m00_2012, 4), m22_50225));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Upper> {m02_3103}, m00_2102, 4), m22_52025));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Lower> {m00_3013}, m00_2012, 4), m22_50225));

  auto m22_2002 = make_native_matrix<M22>(2, 0, 0, 2);
  auto m20_2002 = M20 {m22_2002};
  auto m02_2002 = M02 {m22_2002};
  auto m00_2002 = M00 {m22_2002};

  EXPECT_TRUE(is_near(rank_update(d22_3, m22_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, m22_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, m22_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, m22_2002, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d22_3, m20_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, m20_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, m20_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, m20_2002, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d22_3, m02_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, m02_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, m02_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, m02_2002, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d22_3, m00_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_2_3, m00_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d02_2_3, m00_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_22_3, m00_2002, 4), m22_5005));

  auto m22_501626 = make_native_matrix<M22>(5., 0, 1.6, std::sqrt(26.44));

  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m21_33}, m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m20_33}, m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m01_33}, m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4), m22_501626));

  auto m21_37 = make_native_matrix<M21>(3, 7);
  auto m20_37 = make_native_matrix<M20>(3, 7);
  auto m01_37 = make_native_matrix<M01>(3, 7);
  auto m00_37 = make_native_matrix<M00>(3, 7);

  auto d21_37 = Eigen::DiagonalWrapper {m21_37};
  auto d20_37 = Eigen::DiagonalWrapper {m20_37};
  auto d01_37 = Eigen::DiagonalWrapper {m01_37};
  auto d00_37 = Eigen::DiagonalWrapper {m00_37};

  auto m21_16 = make_native_matrix<M21>(1, 6);
  auto d21_16 = Eigen::DiagonalMatrix<double, 2> {m21_16};
  const auto m21_525 = make_native_matrix<M21>(5, 25);

  EXPECT_TRUE(is_near(rank_update(d21_37, d21_16, 4), Eigen::DiagonalWrapper {m21_525}));
  EXPECT_TRUE(is_near(rank_update(d20_37, d21_16, 4), Eigen::DiagonalWrapper {m21_525}));
  EXPECT_TRUE(is_near(rank_update(d01_37, d21_16, 4), Eigen::DiagonalWrapper {m21_525}));
  EXPECT_TRUE(is_near(rank_update(d00_37, d21_16, 4), Eigen::DiagonalWrapper {m21_525}));
  EXPECT_TRUE(is_near(m21_37, m21_525));
  EXPECT_TRUE(is_near(m20_37, m21_525));
  EXPECT_TRUE(is_near(m01_37, m21_525));
  EXPECT_TRUE(is_near(m00_37, m21_525));


  auto m23_56 = make_native_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);
  auto m23_445 = make_native_matrix<double, 2, 3>(-4, -6, -8, 4.5, 6.5, 8.5);
  M20 m20_3_56 {2,3}; m20_3_56 = m23_56;
  M03 m03_2_56 {2,3}; m03_2_56 = m23_56;
  M00 m00_23_56 {2,3}; m00_23_56 = m23_56;
  EXPECT_TRUE(is_near(solve(m22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m00_23_56), m23_445));

  auto m11_6 = make_native_matrix<M11>(6);
  auto m10_1_6 = M10 {m11_6};
  auto m01_1_6 = M01 {m11_6};
  auto m00_11_6 = M00 {m11_6};

  EXPECT_TRUE(is_near(solve(m11_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, m00_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m00_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m00_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m00_11_6), m11_3));

  auto m12_68 = make_native_matrix<M12>(6, 8);
  auto m10_2_68 = M10 {m12_68};
  auto m02_1_68 = M02 {m12_68};
  auto m00_12_68 = M00 {m12_68};

  auto m12_34 = make_native_matrix<M12>(3, 4);

  EXPECT_TRUE(is_near(solve(m11_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m00_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m00_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m00_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m00_12_68), m12_34));

  auto m11_0 = make_native_matrix<M11>(0);

  EXPECT_TRUE(is_near(solve(m11_0, m12_68), M12::Zero()));
  EXPECT_TRUE(is_near(solve(M10 {m11_0}, m12_68), M12::Zero()));
  EXPECT_TRUE(is_near(solve(M01 {m11_0}, m12_68), M12::Zero()));
  EXPECT_TRUE(is_near(solve(M00 {m11_0}, m12_68), M12::Zero()));

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

  auto m21_25 = make_native_matrix<double, 2, 1>(2, 5);

  EXPECT_TRUE(is_near(reduce_columns(m21), m21));
  EXPECT_TRUE(is_near(reduce_columns(m23), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(m03_2), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(m20_3), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(m00_23), m21_25));
  EXPECT_TRUE(is_near(reduce_columns(cm23), make_native_matrix<cdouble, 2, 1>(cdouble {2,5}, cdouble {5,2})));

  EXPECT_TRUE(is_near(reduce_columns(z22), z21)); static_assert(zero_matrix<decltype(reduce_columns(z22))>);
  EXPECT_TRUE(is_near(reduce_columns(z20_2), z21)); static_assert(zero_matrix<decltype(reduce_columns(z20_2))>);
  EXPECT_TRUE(is_near(reduce_columns(z02_2), z21)); static_assert(zero_matrix<decltype(reduce_columns(z02_2))>);
  EXPECT_TRUE(is_near(reduce_columns(z00_22), z21)); static_assert(zero_matrix<decltype(reduce_columns(z00_22))>);

  EXPECT_TRUE(is_near(reduce_columns(c22_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(c22_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(c20_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(c20_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(c02_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(c02_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(c00_22_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(c00_22_2))> == 2);

  EXPECT_TRUE(is_near(reduce_columns(i22), c21_1)); static_assert(constant_coefficient_v<decltype(reduce_columns(i22))> == 1);
  EXPECT_TRUE(is_near(reduce_columns(i20_2), c21_1)); static_assert(constant_coefficient_v<decltype(reduce_columns(i20_2))> == 1);
  EXPECT_TRUE(is_near(reduce_columns(i02_2), c21_1)); static_assert(constant_coefficient_v<decltype(reduce_columns(i02_2))> == 1);
  EXPECT_TRUE(is_near(reduce_columns(i00_22), c21_1)); static_assert(constant_coefficient_v<decltype(reduce_columns(i00_22))> == 1);

  EXPECT_TRUE(is_near(reduce_columns(d22_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(d22_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(d20_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(d20_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(d02_2_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(d02_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_columns(d00_22_2), c21_2)); static_assert(constant_coefficient_v<decltype(reduce_columns(d00_22_2))> == 2);

  auto m13_234 = make_native_matrix<double, 1, 3>(2.5, 3.5, 4.5);
  EXPECT_TRUE(is_near(reduce_rows(make_native_matrix<double, 1, 3>(1, 2, 3)), make_native_matrix<double, 1, 3>(1, 2, 3)));
  EXPECT_TRUE(is_near(reduce_rows(m23), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(m03_2), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(m20_3), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(m00_23), m13_234));
  EXPECT_TRUE(is_near(reduce_rows(cm23), make_native_matrix<cdouble, 1, 3>(cdouble {2.5,4.5}, cdouble{3.5,3.5}, cdouble {4.5,2.5})));

  EXPECT_TRUE(is_near(reduce_rows(z22), z12)); static_assert(zero_matrix<decltype(reduce_rows(z22))>);
  EXPECT_TRUE(is_near(reduce_rows(z20_2), z12)); static_assert(zero_matrix<decltype(reduce_rows(z20_2))>);
  EXPECT_TRUE(is_near(reduce_rows(z02_2), z12)); static_assert(zero_matrix<decltype(reduce_rows(z02_2))>);
  EXPECT_TRUE(is_near(reduce_rows(z00_22), z12)); static_assert(zero_matrix<decltype(reduce_rows(z00_22))>);

  EXPECT_TRUE(is_near(reduce_rows(c22_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(c22_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(c20_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(c20_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(c02_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(c02_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(c00_22_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(c00_22_2))> == 2);

  EXPECT_TRUE(is_near(reduce_rows(i22), c12_1)); static_assert(constant_coefficient_v<decltype(reduce_rows(i22))> == 1);
  EXPECT_TRUE(is_near(reduce_rows(i20_2), c12_1)); static_assert(constant_coefficient_v<decltype(reduce_rows(i20_2))> == 1);
  EXPECT_TRUE(is_near(reduce_rows(i02_2), c12_1)); static_assert(constant_coefficient_v<decltype(reduce_rows(i02_2))> == 1);
  EXPECT_TRUE(is_near(reduce_rows(i00_22), c12_1)); static_assert(constant_coefficient_v<decltype(reduce_rows(i00_22))> == 1);

  EXPECT_TRUE(is_near(reduce_rows(d22_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(d22_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(d20_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(d20_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(d02_2_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(d02_2_2))> == 2);
  EXPECT_TRUE(is_near(reduce_rows(d00_22_2), c12_2)); static_assert(constant_coefficient_v<decltype(reduce_rows(d00_22_2))> == 2);

  M22 m22_lq_decomp = make_native_matrix<double, 2, 2>(-0.1, 0, 1.096, -1.272);
  EXPECT_TRUE(is_near(LQ_decomposition(make_native_matrix<double, 2, 2>(0.06, 0.08, 0.36, -1.640)), m22_lq_decomp));
  M20 m20_2_lq_decomp {2,2}; m20_2_lq_decomp << 0.06, 0.08, 0.36, -1.640;
  EXPECT_TRUE(is_near(LQ_decomposition(m20_2_lq_decomp), m22_lq_decomp));
  M02 m02_2_lq_decomp {2,2}; m02_2_lq_decomp << 0.06, 0.08, 0.36, -1.640;
  EXPECT_TRUE(is_near(LQ_decomposition(m02_2_lq_decomp), m22_lq_decomp));
  M00 m00_22_lq_decomp {2,2}; m00_22_lq_decomp << 0.06, 0.08, 0.36, -1.640;
  EXPECT_TRUE(is_near(LQ_decomposition(m00_22_lq_decomp), m22_lq_decomp));

  M22 m22_qr_decomp = make_native_matrix<double, 2, 2>(-0.1, 1.096, 0, -1.272);
  EXPECT_TRUE(is_near(QR_decomposition(make_native_matrix<double, 2, 2>(0.06, 0.36, 0.08, -1.640)), m22_qr_decomp));
  M02 m02_2_qr_decomp {2,2}; m02_2_qr_decomp << 0.06, 0.36, 0.08, -1.640;
  EXPECT_TRUE(is_near(QR_decomposition(m02_2_qr_decomp), m22_qr_decomp));
  M20 m20_2_qr_decomp {2,2}; m20_2_qr_decomp << 0.06, 0.36, 0.08, -1.640;
  EXPECT_TRUE(is_near(QR_decomposition(m20_2_qr_decomp), m22_qr_decomp));
  M00 m00_22_qr_decomp {2,2}; m00_22_qr_decomp << 0.06, 0.36, 0.08, -1.640;
  EXPECT_TRUE(is_near(QR_decomposition(m00_22_qr_decomp), m22_qr_decomp));

  EXPECT_TRUE(is_near(LQ_decomposition(m23), adjoint(QR_decomposition(m32))));
  EXPECT_TRUE(is_near(LQ_decomposition(m32), adjoint(QR_decomposition(m23))));
  EXPECT_TRUE(is_near(LQ_decomposition(cm23), adjoint(QR_decomposition(cm32conj))));
  EXPECT_TRUE(is_near(LQ_decomposition(cm32conj), adjoint(QR_decomposition(cm23))));
}


TEST(eigen3, Eigen_Matrix_randomize)
{
  using N = std::normal_distribution<double>;

  M22 m22, m22_true;
  M23 m23, m23_true;
  M32 m32, m32_true;
  M20 m20_2 {2, 2};
  M20 m20_3 {2, 3};
  M02 m02_2 {2, 2};
  M02 m02_3 {2, 2};
  M30 m30 {3, 2};
  M03 m03 {2, 3};
  M00 m00 {2, 2};

  // Test just using the parameters, rather than a constructed distribution.
  m22 = randomize<M22>(N {0.0, 0.7});
  m20_2 = randomize<M20>(2, 2, N {0.0, 1.0});
  m20_3 = randomize<M20>(2, 3, N {0.0, 0.7});
  m02_2 = randomize<M02>(2, 2, N {0.0, 1.0});
  m02_3 = randomize<M02>(3, 2, N {0.0, 0.7});
  m00 = randomize<M00>(2, 2, N {0.0, 1.0});

  // Single distribution for the entire matrix.
  m22 = M22::Zero();
  m20_2 = M20::Zero(2, 2);
  m02_2 = M02::Zero(2, 2);
  m00 = M00::Zero(2, 2);
  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3})) / (i + 1);
    m20_2 = (m20_2 * i + randomize<M20>(2, 2, N {1.0, 0.3})) / (i + 1);
    m02_2 = (m02_2 * i + randomize<M02>(2, 2, N {1.0, 0.3})) / (i + 1);
    m00 = (m00 * i + randomize<M00>(2, 2, N {1.0, 0.3})) / (i + 1);
  }
  m22_true = M22::Constant(1);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m20_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m20_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m02_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m02_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m00, m22_true, 0.1));
  EXPECT_FALSE(is_near(m00, m22_true, 1e-8));

  // A distribution for each element.
  m22 = M22::Zero();
  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})) / (i + 1);
  }
  m22_true = MatrixTraits<M22>::make(1, 2, 3, 4);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));

  // One distribution for each row.
  m32 = M32::Zero();
  m22 = M22::Zero();
  for (int i=0; i<100; i++)
  {
    m32 = (m32 * i + randomize<M32>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
  }
  m32_true = MatrixTraits<M32>::make(1, 1, 2, 2, 3, 3);
  m22_true = MatrixTraits<M22>::make(1, 1, 2, 2);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));

  // One distribution for each column.
  m32 = M32::Zero();
  m23 = M23::Zero();
  for (int i=0; i<100; i++)
  {
    m32 = (m32 * i + randomize<M32>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    m23 = (m23 * i + randomize<M23>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
  }
  m32_true = MatrixTraits<M32>::make(1, 2, 1, 2, 1, 2);
  m23_true = MatrixTraits<M23>::make(1, 2, 3, 1, 2, 3);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m23, m23_true, 0.1));
  EXPECT_FALSE(is_near(m23, m23_true, 1e-8));
}


TEST(eigen3, Matrix_blocks)
{
  auto m21 = make_native_matrix<M21>(1, 2);
  auto m12 = make_native_matrix<M12>(1, 2);
  auto m22 = make_native_matrix<M22>(1, 2, 3, 4);
  auto m32 = make_native_matrix<M32>(1, 2, 3, 4, 5, 6);
  auto m23 = make_native_matrix<M23>(1, 2, 3, 4, 5, 6);

  M22 el22 {m22}; // 1, 2, 3, 4
  M20 el20_2 {m22};
  M02 el02_2 {m22};
  M00 el00_22 {m22};

  EXPECT_NEAR(get_element(el22, 0, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el20_2, 0, 1), 2, 1e-8);
  EXPECT_NEAR(get_element(el02_2, 1, 0), 3, 1e-8);
  EXPECT_NEAR(get_element(el00_22, 1, 1), 4, 1e-8);

  set_element(el22, 5.5, 1, 0); EXPECT_NEAR(get_element(el22, 1, 0), 5.5, 1e-8);
  set_element(el20_2, 5.5, 1, 0); EXPECT_NEAR(get_element(el20_2, 1, 0), 5.5, 1e-8);
  set_element(el02_2, 5.5, 1, 0); EXPECT_NEAR(get_element(el02_2, 1, 0), 5.5, 1e-8);
  set_element(el00_22, 5.5, 1, 0); EXPECT_NEAR(get_element(el00_22, 1, 0), 5.5, 1e-8);

  M21 el21 {m21}; // 1, 2
  M20 el20_1 {m21};
  M01 el01_2 {m21};
  M00 el00_21 {m21};

  EXPECT_NEAR(get_element(el21, 1, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el20_1, 2, 0), 2, 1e-8);
  EXPECT_NEAR(get_element(el01_2, 1, 0), 1, 1e-8);
  EXPECT_NEAR(get_element(el00_21, 2, 0), 2, 1e-8);

  set_element(el21, 5.5, 1, 0); EXPECT_NEAR(get_element(el21, 1, 0), 5.5, 1e-8);
  set_element(el20_1, 5.5, 1, 0); EXPECT_NEAR(get_element(el20_1, 1, 0), 5.5, 1e-8);
  set_element(el01_2, 5.5, 1, 0); EXPECT_NEAR(get_element(el01_2, 1, 0), 5.5, 1e-8);
  set_element(el00_21, 5.5, 1, 0); EXPECT_NEAR(get_element(el00_21, 1, 0), 5.5, 1e-8);

  set_element(el21, 5.6, 1); EXPECT_NEAR(get_element(el21, 1), 5.6, 1e-8);
  set_element(el01_2, 5.6, 1); EXPECT_NEAR(get_element(el01_2, 1), 5.6, 1e-8);

  M12 el12 {m12}; // 1, 2
  M10 el10_2 {m12};
  M02 el02_1 {m12};
  M00 el00_12 {m12};

  EXPECT_NEAR(get_element(el12, 0, 1), 1, 1e-8);
  EXPECT_NEAR(get_element(el10_2, 0, 2), 2, 1e-8);
  EXPECT_NEAR(get_element(el02_1, 0, 1), 1, 1e-8);
  EXPECT_NEAR(get_element(el00_12, 0, 2), 2, 1e-8);

  set_element(el12, 5.5, 0, 1); EXPECT_NEAR(get_element(el12, 0, 1), 5.5, 1e-8);
  set_element(el10_2, 5.5, 0, 1); EXPECT_NEAR(get_element(el10_2, 0, 1), 5.5, 1e-8);
  set_element(el02_1, 5.5, 0, 1); EXPECT_NEAR(get_element(el02_1, 0, 1), 5.5, 1e-8);
  set_element(el00_12, 5.5, 0, 1); EXPECT_NEAR(get_element(el00_12, 0, 1), 5.5, 1e-8);

  set_element(el12, 5.6, 1); EXPECT_NEAR(get_element(el12, 1), 5.6, 1e-8);
  set_element(el10_2, 5.6, 1); EXPECT_NEAR(get_element(el10_2, 1), 5.6, 1e-8);

  auto m12_56 = make_native_matrix<M12>(5, 6);

  EXPECT_TRUE(is_near(concatenate_vertical(m22, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M20 {m22}, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M20 {m22}, M10 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M20 {m22}, M02 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M20 {m22}, M00 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M02 {m22}, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M02 {m22}, M10 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M02 {m22}, M02 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M02 {m22}, M00 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M00 {m22}, m12_56), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M00 {m22}, M10 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M00 {m22}, M02 {m12_56}), m32));
  EXPECT_TRUE(is_near(concatenate_vertical(M00 {m22}, M00 {m12_56}), m32));

  auto m21_56 = make_native_matrix<M21>(5, 6);

  EXPECT_TRUE(is_near(concatenate_horizontal(m22, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M20 {m22}, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M20 {m22}, M20 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M20 {m22}, M01 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M20 {m22}, M00 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M02 {m22}, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M02 {m22}, M20 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M02 {m22}, M01 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M02 {m22}, M00 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M00 {m22}, m21_56), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M00 {m22}, M20 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M00 {m22}, M01 {m21_56}), m23));
  EXPECT_TRUE(is_near(concatenate_horizontal(M00 {m22}, M00 {m21_56}), m23));

  auto m22_5678 = make_native_matrix<double, 2, 2>(5, 6, 7, 8);
  auto m44_diag = make_native_matrix<double, 4, 4>(1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8);

  EXPECT_TRUE(is_near(concatenate_diagonal(m22, m22_5678), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M20 {m22}, m22_5678), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M20 {m22}, M20 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M20 {m22}, M02 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M20 {m22}, M00 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M02 {m22}, m22_5678), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M02 {m22}, M20 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M02 {m22}, M02 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M02 {m22}, M00 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M00 {m22}, m22_5678), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M00 {m22}, M20 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M00 {m22}, M02 {m22_5678}), m44_diag));
  EXPECT_TRUE(is_near(concatenate_diagonal(M00 {m22}, M00 {m22_5678}), m44_diag));

  EXPECT_TRUE(is_near(split_vertical(make_native_matrix<M22>(1, 0, 0, 2)), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical(M20 {make_native_matrix<M22>(1, 0, 0, 2)}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical(M02 {make_native_matrix<M22>(1, 0, 0, 2)}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical(M00 {make_native_matrix<M22>(1, 0, 0, 2)}), std::tuple {}));

  auto x1 = make_native_matrix<double, 5, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3,
    4, 0, 0,
    0, 5, 0);

  auto m33 = make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto tup_m33_m23 = std::tuple {m33, make_native_matrix<double, 2, 3>(
    4, 0, 0,
    0, 5, 0)};

  auto tup_m23_m23 = std::tuple {make_native_matrix<double, 2, 3>(
    1, 0, 0,
    0, 2, 0), make_native_matrix<double, 2, 3>(
    0, 0, 3,
    4, 0, 0)};

  EXPECT_TRUE(is_near(split_vertical<3, 2>(x1), tup_m33_m23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(eigen_matrix_t<double, 5, 3> {x1}), tup_m33_m23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(eigen_matrix_t<double, 5, 0> {x1}), tup_m33_m23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(eigen_matrix_t<double, 0, 3> {x1}), tup_m33_m23));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(eigen_matrix_t<double, 0, 0> {x1}), tup_m33_m23));

  EXPECT_TRUE(is_near(split_vertical<2, 2>(x1), tup_m23_m23));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(eigen_matrix_t<double, 5, 3> {x1}), tup_m23_m23));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(eigen_matrix_t<double, 5, 0> {x1}), tup_m23_m23));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(eigen_matrix_t<double, 0, 3> {x1}), tup_m23_m23));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(eigen_matrix_t<double, 0, 0> {x1}), tup_m23_m23));

  auto m22_12 = make_native_matrix<double, 2, 2>(
    1, 0,
    0, 2);

  EXPECT_TRUE(is_near(split_horizontal(m22_12), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(M22 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(M20 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(M02 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(M00 {m22_12}), std::tuple {}));

  const auto b1 = make_native_matrix<double, 3, 5>(
    1, 0, 0, 0, 0,
    0, 2, 0, 4, 0,
    0, 0, 3, 0, 5);

  auto tup_m33_m32 = std::tuple {m33, make_native_matrix<double, 3, 2>(
    0, 0,
    4, 0,
    0, 5)};

  EXPECT_TRUE(is_near(split_horizontal<3, 2>(b1), tup_m33_m32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(eigen_matrix_t<double, 3, 5> {b1}), tup_m33_m32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(eigen_matrix_t<double, 3, 0> {b1}), tup_m33_m32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(eigen_matrix_t<double, 0, 5> {b1}), tup_m33_m32));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(eigen_matrix_t<double, 0, 0> {b1}), tup_m33_m32));

  auto tup_m32_m32 = std::tuple {make_native_matrix<double, 3, 2>(
    1, 0,
    0, 2,
    0, 0), make_native_matrix<double, 3, 2>(
    0, 0,
    0, 4,
    3, 0)};

  EXPECT_TRUE(is_near(split_horizontal<2, 2>(b1), tup_m32_m32));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(eigen_matrix_t<double, 3, 5>(b1)), tup_m32_m32));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(eigen_matrix_t<double, 3, 0>(b1)), tup_m32_m32));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(eigen_matrix_t<double, 0, 5>(b1)), tup_m32_m32));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(eigen_matrix_t<double, 0, 0>(b1)), tup_m32_m32));

  EXPECT_TRUE(is_near(split_diagonal(m22_12), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal(M22 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal(M20 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal(M02 {m22_12}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal(M00 {m22_12}), std::tuple {}));

  auto m55 = make_native_matrix<double, 5, 5>(
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 3, 0, 0,
    0, 0, 0, 4, 0,
    0, 0, 0, 0, 5);

  auto tup_m33_m22 = std::tuple {m33, make_native_matrix<double, 2, 2>(
    4, 0,
    0, 5)};

  EXPECT_TRUE(is_near(split_diagonal<3, 2>(m55), tup_m33_m22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(M55 {m55}), tup_m33_m22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(M50 {m55}), tup_m33_m22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(M05 {m55}), tup_m33_m22));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(M00 {m55}), tup_m33_m22));

  auto tup_m22_m22 = std::tuple {m22, make_native_matrix<double, 2, 2>(
    3, 0,
    0, 4)};

  EXPECT_TRUE(is_near(split_diagonal<2, 2>(m55), tup_m22_m22));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(M55 {m55}), tup_m22_m22));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(M50 {m55}), tup_m22_m22));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(M05 {m55}), tup_m22_m22));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(M00 {m55}), tup_m22_m22));

  EXPECT_TRUE(is_near(column(m33, 2), make_native_matrix<M31>(0., 0, 3)));
  EXPECT_TRUE(is_near(column(M30 {m33}, 2), make_native_matrix<M31>(0., 0, 3)));
  EXPECT_TRUE(is_near(column(M03 {m33}, 2), make_native_matrix<M31>(0., 0, 3)));
  EXPECT_TRUE(is_near(column(M00 {m33}, 2), make_native_matrix<M31>(0., 0, 3)));

  EXPECT_TRUE(is_near(column<1>(m33), make_native_matrix<M31>(0., 2, 0)));
  EXPECT_TRUE(is_near(column<1>(M30 {m33}), make_native_matrix<M31>(0., 2, 0)));
  EXPECT_TRUE(is_near(column<1>(M03 {m33}), make_native_matrix<M31>(0., 2, 0)));
  EXPECT_TRUE(is_near(column<1>(M00 {m33}), make_native_matrix<M31>(0., 2, 0)));

  EXPECT_TRUE(is_near(row(m33, 2), make_native_matrix<double, 1, 3>(0, 0, 3)));
  EXPECT_TRUE(is_near(row(M30 {m33}, 2), make_native_matrix<M13>(0, 0, 3)));
  EXPECT_TRUE(is_near(row(M03 {m33}, 2), make_native_matrix<M13>(0, 0, 3)));
  EXPECT_TRUE(is_near(row(M00 {m33}, 2), make_native_matrix<M13>(0, 0, 3)));

  EXPECT_TRUE(is_near(row<1>(m33), make_native_matrix<M13>(0, 2, 0)));
  EXPECT_TRUE(is_near(row<1>(M30 {m33}), make_native_matrix<M13>(0, 2, 0)));
  EXPECT_TRUE(is_near(row<1>(M03 {m33}), make_native_matrix<M13>(0, 2, 0)));
  EXPECT_TRUE(is_near(row<1>(M00 {m33}), make_native_matrix<M13>(0, 2, 0)));

  M21 m21_vol {m21};
  M20 m20_1_vol {m21};
  M01 m01_2_vol {m21};
  M00 m00_21_vol {m21};

  auto m21_23 = make_native_matrix<M21>(2, 3);

  apply_columnwise(m21_vol, [](auto& col){ col += col.Constant(1); }); EXPECT_TRUE(is_near(m21_vol, m21_23));
  apply_columnwise(m20_1_vol, [](auto& col){ col += col.Constant(1); }); EXPECT_TRUE(is_near(m20_1_vol, m21_23));
  apply_columnwise(m01_2_vol, [](auto& col){ col += col.Constant(2, 1, 1); }); EXPECT_TRUE(is_near(m01_2_vol, m21_23));
  apply_columnwise(m00_21_vol, [](auto& col){ col += col.Constant(2, 1, 1); }); EXPECT_TRUE(is_near(m00_21_vol, m21_23));

  M33 m33_vol {m33};
  M30 m30_vol {m33};
  M03 m03_vol {m33};
  M00 m00_vol {m33};

  auto m33_234 = make_native_matrix<M33>(
    2, 1, 1,
    1, 3, 1,
    1, 1, 4);

  apply_columnwise(m33_vol, [](auto& col){ col += col.Constant(1); }); EXPECT_TRUE(is_near(m33_vol, m33_234));
  apply_columnwise(m30_vol, [](auto& col){ col += col.Constant(1); }); EXPECT_TRUE(is_near(m30_vol, m33_234));
  apply_columnwise(m03_vol, [](auto& col){ col += col.Constant(3, 1, 1); }); EXPECT_TRUE(is_near(m03_vol, m33_234));
  apply_columnwise(m00_vol, [](auto& col){ col += col.Constant(3, 1, 1); }); EXPECT_TRUE(is_near(m00_vol, m33_234));

  m33_vol = m33; EXPECT_TRUE(is_near(apply_columnwise(m33_vol, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }), m33_234));
  m30_vol = m33; EXPECT_TRUE(is_near(apply_columnwise(m30_vol, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }), m33_234));
  m03_vol = m33; EXPECT_TRUE(is_near(apply_columnwise(m03_vol, [](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }), m33_234));
  m00_vol = m33; EXPECT_TRUE(is_near(apply_columnwise(m00_vol, [](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }), m33_234));

  EXPECT_TRUE(is_near(m33_vol, m33_234));
  EXPECT_TRUE(is_near(m30_vol, m33_234));
  EXPECT_TRUE(is_near(m03_vol, m33_234));
  EXPECT_TRUE(is_near(m00_vol, m33_234));

  EXPECT_TRUE(is_near(apply_columnwise(M33 {m33}, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }), m33_234));
  EXPECT_TRUE(is_near(apply_columnwise(M30 {m33}, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }), m33_234));
  EXPECT_TRUE(is_near(apply_columnwise(M03 {m33}, [](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }), m33_234));
  EXPECT_TRUE(is_near(apply_columnwise(M00 {m33}, [](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }), m33_234));

  auto m33_135_col = make_native_matrix<M33>(
    1, 1, 2,
    0, 3, 2,
    0, 1, 5);

  m33_vol = m33; apply_columnwise(m33_vol, [](auto& col, std::size_t i){ col += col.Constant(i); }); EXPECT_TRUE(is_near(m33_vol, m33_135_col));
  m30_vol = m33; apply_columnwise(m30_vol, [](auto& col, std::size_t i){ col += col.Constant(i); }); EXPECT_TRUE(is_near(m30_vol, m33_135_col));
  m03_vol = m33; apply_columnwise(m03_vol, [](auto& col, std::size_t i){ col += col.Constant(3, 1, i); }); EXPECT_TRUE(is_near(m03_vol, m33_135_col));
  m00_vol = m33; apply_columnwise(m00_vol, [](auto& col, std::size_t i){ col += col.Constant(3, 1, i); }); EXPECT_TRUE(is_near(m00_vol, m33_135_col));

  EXPECT_TRUE(is_near(apply_columnwise(m33, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise(M33 {m33}, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise(M30 {m33}, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise(M03 {m33}, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(3, 1, i)); }), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise(M00 {m33}, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(3, 1, i)); }), m33_135_col));

  auto m33_123_h = make_native_matrix<M33>(
    1, 1, 1,
    2, 2, 2,
    3, 3, 3);

  auto m31_123 = make_native_matrix<M31>(1, 2, 3);

  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return m31_123; }), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return M30 {m31_123}; }), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return M01 {m31_123}; }), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return M00 {m31_123}; }), m33_123_h));

  EXPECT_TRUE(is_near(apply_columnwise([&] { return m31_123; }, 3), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise([&] { return M30 {m31_123}; }, 3), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise([&] { return M01 {m31_123}; }, 3), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise([&] { return M00 {m31_123}; }, 3), m33_123_h));

  auto m33_123_d = make_native_matrix<M33>(
    1, 2, 3,
    2, 3, 4,
    3, 4, 5);

  EXPECT_TRUE(is_near(apply_columnwise<3>([](std::size_t i) { return make_native_matrix<M31>(1 + i, 2 + i, 3 + i); }), m33_123_d));

  EXPECT_TRUE(is_near(apply_columnwise([](std::size_t i) { return make_native_matrix<M31>(1 + i, 2 + i, 3 + i); }, 3), m33_123_d));

  M12 m12_vol {m12};
  M10 m10_2_vol {m12};
  M02 m02_1_vol {m12};
  M00 m00_12_vol {m12};

  auto m12_23 = make_native_matrix<M12>(2, 3);

  apply_rowwise(m12_vol, [](auto& row){ row += row.Constant(1); }); EXPECT_TRUE(is_near(m12_vol, m12_23));
  apply_rowwise(m10_2_vol, [](auto& row){ row += row.Constant(1, 2, 1); }); EXPECT_TRUE(is_near(m10_2_vol, m12_23));
  apply_rowwise(m02_1_vol, [](auto& row){ row += row.Constant(1); }); EXPECT_TRUE(is_near(m02_1_vol, m12_23));
  apply_rowwise(m00_12_vol, [](auto& row){ row += row.Constant(1, 2, 1); }); EXPECT_TRUE(is_near(m00_12_vol, m12_23));

  m33_vol = m33; apply_rowwise(m33_vol, [](auto& row){ row += row.Constant(1); }); EXPECT_TRUE(is_near(m33_vol, m33_234));
  m30_vol = m33; apply_rowwise(m30_vol, [](auto& row){ row += row.Constant(1, 3, 1); }); EXPECT_TRUE(is_near(m30_vol, m33_234));
  m03_vol = m33; apply_rowwise(m03_vol, [](auto& row){ row += row.Constant(1); }); EXPECT_TRUE(is_near(m03_vol, m33_234));
  m00_vol = m33; apply_rowwise(m00_vol, [](auto& row){ row += row.Constant(1, 3, 1); }); EXPECT_TRUE(is_near(m00_vol, m33_234));

  m33_vol = m33; EXPECT_TRUE(is_near(apply_rowwise(m33_vol, [](const auto& row){ return make_self_contained(row + row.Constant(1)); }), m33_234));
  m30_vol = m33; EXPECT_TRUE(is_near(apply_rowwise(m30_vol, [](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }), m33_234));
  m03_vol = m33; EXPECT_TRUE(is_near(apply_rowwise(m03_vol, [](const auto& row){ return make_self_contained(row + row.Constant(1)); }), m33_234));
  m00_vol = m33; EXPECT_TRUE(is_near(apply_rowwise(m00_vol, [](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }), m33_234));

  EXPECT_TRUE(is_near(m33_vol, m33_234));
  EXPECT_TRUE(is_near(m30_vol, m33_234));
  EXPECT_TRUE(is_near(m03_vol, m33_234));
  EXPECT_TRUE(is_near(m00_vol, m33_234));

  EXPECT_TRUE(is_near(apply_rowwise(M33 {m33}, [](const auto& row){ return make_self_contained(row + row.Constant(1)); }), m33_234));
  EXPECT_TRUE(is_near(apply_rowwise(M30 {m33}, [](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }), m33_234));
  EXPECT_TRUE(is_near(apply_rowwise(M03 {m33}, [](const auto& row){ return make_self_contained(row + row.Constant(1)); }), m33_234));
  EXPECT_TRUE(is_near(apply_rowwise(M00 {m33}, [](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }), m33_234));

  auto m33_135_row = make_native_matrix<double, 3, 3>(
    1, 0, 0,
    1, 3, 1,
    2, 2, 5);

  m33_vol = m33; apply_rowwise(m33_vol, [](auto& row, std::size_t i){ row += row.Constant(i); }); EXPECT_TRUE(is_near(m33_vol, m33_135_row));
  m30_vol = m33; apply_rowwise(m30_vol, [](auto& row, std::size_t i){ row += row.Constant(1, 3, i); }); EXPECT_TRUE(is_near(m30_vol, m33_135_row));
  m03_vol = m33; apply_rowwise(m03_vol, [](auto& row, std::size_t i){ row += row.Constant(i); }); EXPECT_TRUE(is_near(m03_vol, m33_135_row));
  m00_vol = m33; apply_rowwise(m00_vol, [](auto& row, std::size_t i){ row += row.Constant(1, 3, i); }); EXPECT_TRUE(is_near(m00_vol, m33_135_row));

  EXPECT_TRUE(is_near(apply_rowwise(m33, [](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise(M33 {m33}, [](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise(M30 {m33}, [](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(1, 3, i)); }), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise(M03 {m33}, [](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise(M00 {m33}, [](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(1, 3, i)); }), m33_135_row));

  auto m33_123_v = make_native_matrix<double, 3, 3>(
    1, 2, 3,
    1, 2, 3,
    1, 2, 3);

  auto m13_123 = make_native_matrix<M13>(1, 2, 3);

  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return m13_123; }), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return M10 {m13_123}; }), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return M03 {m13_123}; }), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return M00 {m13_123}; }), m33_123_v));

  EXPECT_TRUE(is_near(apply_rowwise([&] { return m13_123; }, 3), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise([&] { return M10 {m13_123}; }, 3), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise([&] { return M03 {m13_123}; }, 3), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise([&] { return M00 {m13_123}; }, 3), m33_123_v));

  EXPECT_TRUE(is_near(apply_rowwise<3>([](std::size_t i) { return make_native_matrix<M13>(1 + i, 2 + i, 3 + i); }), m33_123_d));

  EXPECT_TRUE(is_near(apply_rowwise([](std::size_t i) { return make_native_matrix<M13>(1 + i, 2 + i, 3 + i); }, 3), m33_123_d));

  m33_vol = m33; apply_coefficientwise(m33_vol, [](auto& x){ x += 1; }); EXPECT_TRUE(is_near(m33_vol, m33_234));
  m30_vol = m33; apply_coefficientwise(m30_vol, [](auto& x){ x += 1; }); EXPECT_TRUE(is_near(m30_vol, m33_234));
  m03_vol = m33; apply_coefficientwise(m03_vol, [](auto& x){ x += 1; }); EXPECT_TRUE(is_near(m03_vol, m33_234));
  m00_vol = m33; apply_coefficientwise(m00_vol, [](auto& x){ x += 1; }); EXPECT_TRUE(is_near(m00_vol, m33_234));

  EXPECT_TRUE(is_near(apply_coefficientwise(m33, [](double& x){ return x + 1; }), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise(M33 {m33}, [](double& x){ return x + 1; }), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise(M30 {m33}, [](double& x){ return x + 1; }), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise(M03 {m33}, [](double& x){ return x + 1; }), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise(M00 {m33}, [](double& x){ return x + 1; }), m33_234));

  auto m33_147 = make_native_matrix<double, 3, 3>(
    1, 1, 2,
    1, 4, 3,
    2, 3, 7);

  m33_vol = m33; apply_coefficientwise(m33_vol, [](auto& x, std::size_t i, std::size_t j){ x += i + j; }); EXPECT_TRUE(is_near(m33_vol, m33_147));
  m30_vol = m33; apply_coefficientwise(m30_vol, [](auto& x, std::size_t i, std::size_t j){ x += i + j; }); EXPECT_TRUE(is_near(m30_vol, m33_147));
  m03_vol = m33; apply_coefficientwise(m03_vol, [](auto& x, std::size_t i, std::size_t j){ x += i + j; }); EXPECT_TRUE(is_near(m03_vol, m33_147));
  m00_vol = m33; apply_coefficientwise(m00_vol, [](auto& x, std::size_t i, std::size_t j){ x += i + j; }); EXPECT_TRUE(is_near(m00_vol, m33_147));

  EXPECT_TRUE(is_near(apply_coefficientwise(m33, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise(M33 {m33}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise(M30 {m33}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise(M03 {m33}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise(M00 {m33}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), m33_147));
}

