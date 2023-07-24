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

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;


TEST(eigen3, cwise_nullary_operations)
{
  static_assert(native_eigen_matrix<M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::IdentityReturnType>);
  static_assert(self_contained<const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>);

  static_assert(constant_coefficient_v<C11_1> == 1);
  static_assert(constant_matrix<typename Mxx::ConstantReturnType, CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<typename Mxx::ConstantReturnType, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_coefficient_v<Z11> == 0);
  static_assert(constant_coefficient_v<Z22> == 0);

  EXPECT_EQ(constant_coefficient{M22::Constant(3)}(), 3);
  EXPECT_EQ(constant_coefficient{M2x::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{Mx2::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{Mxx::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{M11::Identity()}(), 1);
  EXPECT_EQ(constant_coefficient{M1x::Identity(1, 1)}(), 1);
  EXPECT_EQ(constant_coefficient{Mx1::Identity(1, 1)}(), 1);
  EXPECT_EQ(constant_coefficient{Mxx::Identity(1, 1)}(), 1);

  static_assert(not zero_matrix<typename Mxx::ConstantReturnType>);
  static_assert(zero_matrix<Z11>);
  static_assert(zero_matrix<Z22>);

  static_assert(constant_diagonal_coefficient_v<I11> == 1);
  static_assert(constant_diagonal_coefficient_v<I1x> == 1);
  static_assert(constant_diagonal_coefficient_v<Ix1> == 1);
  static_assert(constant_diagonal_coefficient_v<Ixx> == 1);

  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<I22> == 1);
  static_assert(not constant_diagonal_matrix<typename Mxx::ConstantReturnType, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<typename Mxx::ConstantReturnType, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<typename Mxx::ConstantReturnType, CompileTimeStatus::unknown>);
  static_assert(constant_diagonal_coefficient_v<Z11> == 0);
  static_assert(constant_diagonal_coefficient_v<Z22> == 0);

  static_assert(constant_diagonal_matrix<typename M33::IdentityReturnType>);
  static_assert(constant_diagonal_matrix<typename M3x::IdentityReturnType, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<typename Mx3::IdentityReturnType, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<typename Mxx::IdentityReturnType, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<typename M3x::IdentityReturnType>);
  static_assert(not constant_diagonal_matrix<typename Mx3::IdentityReturnType>);
  static_assert(not constant_diagonal_matrix<typename Mxx::IdentityReturnType>);

  EXPECT_EQ(constant_diagonal_coefficient{M22::Identity()}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M2x::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{Mx2::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{Mxx::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M11::Constant(3)}(), 3);
  EXPECT_EQ(constant_diagonal_coefficient{M1x::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_diagonal_coefficient{Mx1::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_diagonal_coefficient{Mxx::Constant(1, 1, 3)}(), 3);

  static_assert(identity_matrix<typename M33::IdentityReturnType>);
  static_assert(identity_matrix<typename M3x::IdentityReturnType, Likelihood::maybe>);
  static_assert(identity_matrix<typename Mx3::IdentityReturnType, Likelihood::maybe>);
  static_assert(identity_matrix<typename Mxx::IdentityReturnType, Likelihood::maybe>);
  static_assert(not identity_matrix<typename M3x::IdentityReturnType>);
  static_assert(not identity_matrix<typename Mx3::IdentityReturnType>);
  static_assert(not identity_matrix<typename Mxx::IdentityReturnType>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<Z22>);

  static_assert(diagonal_matrix<typename M33::IdentityReturnType>);

  static_assert(hermitian_matrix<M33::ConstantReturnType>);

  static_assert(hermitian_matrix<M33::ConstantReturnType>);
  static_assert(not hermitian_matrix<M21::ConstantReturnType>);
  static_assert(hermitian_matrix<typename M33::IdentityReturnType>);
  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<C11_2>);

  static_assert(triangular_matrix<Z22, TriangleType::lower>);

  static_assert(triangular_matrix<Z22, TriangleType::upper>);

  static_assert(square_matrix<Z11>);
  static_assert(square_matrix<C11_1>);

  static_assert(square_matrix<Z11, Likelihood::maybe>);
  static_assert(square_matrix<Z2x, Likelihood::maybe>);
  static_assert(not square_matrix<Z21, Likelihood::maybe>);
  static_assert(square_matrix<C22_1, Likelihood::maybe>);
  static_assert(not square_matrix<C21_1, Likelihood::maybe>);

  static_assert(one_by_one_matrix<Z11>);
  static_assert(one_by_one_matrix<Z1x, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Zx1, Likelihood::maybe>);
  static_assert(one_by_one_matrix<C11_1>);

  static_assert(not writable<Mx2::ConstantReturnType>);
  static_assert(not writable<M2x::IdentityReturnType>);

  static_assert(not modifiable<M33::ConstantReturnType, M33>);
  static_assert(not modifiable<M33::IdentityReturnType, M33>);
  static_assert(not modifiable<M33::ConstantReturnType, M33::ConstantReturnType>);
  static_assert(not modifiable<M33::IdentityReturnType, M33::IdentityReturnType>);
}


TEST(eigen3, cwise_unary_operations)
{
  auto id = I22 {2, 2}; // Identity
  auto zero = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2

  static_assert(self_contained<const M22>);

  // scalar_opposite_op
  static_assert(constant_coefficient_v<decltype(-std::declval<C11_2>())> == -2);
  static_assert(constant_coefficient_v<decltype(-std::declval<C22_2>())> == -2);
  EXPECT_EQ((constant_coefficient{-cp2}()), -2);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<C11_2>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<I22>())> == -1);
  EXPECT_EQ((constant_diagonal_coefficient{-cdm2}()), 2);
  static_assert(not constant_matrix<decltype(-std::declval<Cd22_m1>()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_matrix<decltype(-std::declval<Ixx>()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_matrix<decltype(-std::declval<Cxx_m1>()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<decltype(-std::declval<Cdxx_m1>()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(zero_matrix<decltype(-zero)>);
  static_assert(identity_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(triangular_matrix<decltype(-std::declval<Tlv22>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(-std::declval<Tuv22>()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(-zero)>);
  static_assert(diagonal_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<I22>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<I2x>()), Likelihood::maybe>);
  static_assert(hermitian_matrix<decltype(-std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(-std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(-zero)>);
  static_assert(not hermitian_matrix<decltype(cxa)>);
  static_assert(not hermitian_matrix<decltype(-cxb)>);
  static_assert(not writable<decltype(-std::declval<M22>())>);
  static_assert(not modifiable<decltype(-std::declval<M33>()), M33>);

  // scalar_abs_op
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().abs())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().abs())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().abs())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().abs())> == 1);
  static_assert(not constant_matrix<decltype(std::declval<Cd22_m1>().abs()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_matrix<decltype(std::declval<Cxx_m1>().abs()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<decltype(std::declval<Cdxx_m1>().abs()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(identity_matrix<decltype((-id).abs())>);
  static_assert(identity_matrix<decltype(std::declval<C11_m1>().abs())>);
  static_assert(zero_matrix<decltype(zero.abs())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cwiseAbs()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).abs())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cwiseAbs())>);
  static_assert(hermitian_matrix<decltype(cxa.abs())>);

  // scalar_score_coeff_op (inherits from scalar_abs_op)
  static_assert(constant_diagonal_coefficient_v<Eigen::CwiseUnaryOp<Eigen::internal::scalar_score_coeff_op<double>, Cd22_m2>> == 2);

  // abs_knowing_score not implemented because it is not a true Eigen functor.

  // scalar_abs2_op
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().abs2())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<Cxx_m2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cdxx_2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().abs2())> == 1);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cdxx_m1>().abs2())> == 1);
  static_assert(not constant_matrix<decltype(std::declval<Cd22_m1>().abs2()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_matrix<decltype(std::declval<Cxx_m1>().abs2()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<decltype(std::declval<Cdxx_m1>().abs2()), CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(identity_matrix<decltype((-id).abs2())>);
  static_assert(identity_matrix<decltype(std::declval<C11_m1>().abs2())>);
  static_assert(zero_matrix<decltype(zero.abs2())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cwiseAbs2()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).abs2())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cwiseAbs2())>);
  static_assert(hermitian_matrix<decltype(cxa.abs2())>);

  // scalar_conjugate_op
  static_assert(constant_coefficient_v<decltype(M11::Identity().conjugate())> == 1);
  EXPECT_EQ((constant_coefficient{cxa.conjugate()}()), (std::complex<double>{1, -2}));
  EXPECT_EQ((constant_coefficient{cxb.conjugate()}()), (std::complex<double>{3, -4}));
  static_assert(constant_diagonal_coefficient_v<decltype(M11::Identity().conjugate())> == 1);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().conjugate())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().conjugate()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().conjugate()), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().conjugate())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().conjugate())>);
  static_assert(hermitian_matrix<decltype(std::declval<I22>().conjugate())>);

  // scalar_arg_op
  EXPECT_EQ(constant_coefficient{cp2.arg()}(), std::arg(2));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().arg())> == 0);
  EXPECT_EQ(constant_coefficient{cm2.arg()}(), std::arg(-2));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().arg())> == pi);
  EXPECT_TRUE(are_within_tolerance<10>((constant_coefficient{cxa.arg()}()), (std::arg(std::complex<double>{1, 2}))));
  EXPECT_TRUE(are_within_tolerance<10>((constant_coefficient{cxb.arg()}()), (std::arg(std::complex<double>{3, 4}))));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().arg())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().cwiseAbs()), TriangleType::lower>);
  static_assert(not diagonal_matrix<decltype(std::declval<Cd22_2>().arg())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cwiseAbs()), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cwiseAbs())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().cwiseAbs())>);
  static_assert(hermitian_matrix<decltype(cxa.arg())>);

  // scalar_cast_op
  using SCOp = Eigen::internal::scalar_cast_op<double, int>;
  static_assert(std::is_same_v<typename constant_coefficient<Eigen::CwiseUnaryOp<SCOp, C22_2>>::value_type, int>);
  static_assert(std::is_same_v<typename constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cast_op<std::complex<double>, std::complex<int>>, decltype(cxa)>>::value_type, std::complex<int>>);
  static_assert(constant_diagonal_matrix<Eigen::CwiseUnaryOp<SCOp, Cd22_2>>);
  static_assert(triangular_matrix<Eigen::CwiseUnaryOp<SCOp, Tlv22>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::CwiseUnaryOp<SCOp, Tuv22>, TriangleType::upper>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<SCOp, Cd22_2>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<SCOp, Salv22>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<SCOp, Sauv22>>);
  static_assert(not hermitian_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cast_op<std::complex<double>, std::complex<int>>, decltype(cxa)>>);

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  auto id1_int = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<int>, Eigen::Array<int, 1, 1>> {1, 1}; // Identity
  auto id2_int = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<int>, Eigen::Array<int, 2, 2>> {2, 2}; // Identity
  auto cp2_int = (id1_int + id1_int).replicate<2, 2>(); // Constant +2
  auto cdp2_int = id2_int * 2; // Constant diagonal +2

  // scalar_shift_right_op
  EXPECT_EQ(constant_coefficient{cp2_int.shiftRight<1>()}(), cp2_int.shiftRight<1>()(0, 0));
  EXPECT_EQ(constant_coefficient{cp2_int.shiftRight<1>()}(), cp2_int.shiftRight<1>()(0, 1));
  static_assert(constant_coefficient_v<decltype(cp2_int.shiftRight<1>())> == 1);
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftRight<1>()}(), cdp2_int.shiftRight<1>()(0, 0));
  EXPECT_EQ(0, cdp2_int.shiftRight<1>()(0, 1));
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftRight<1>()}(), 1);
  static_assert(constant_diagonal_matrix<decltype(cdp2_int.shiftRight<1>()), CompileTimeStatus::unknown>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().cast<int>().array().shiftRight<1>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cast<int>().array().shiftRight<1>()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().cast<int>().shiftRight<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cast<int>().array().shiftRight<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().cast<int>().array().shiftRight<1>())>);

  // scalar_shift_left_op
  EXPECT_EQ(constant_coefficient{cp2_int.shiftLeft<1>()}(), cp2_int.shiftLeft<1>()(0, 0));
  EXPECT_EQ(constant_coefficient{cp2_int.shiftLeft<1>()}(), cp2_int.shiftLeft<1>()(0, 1));
  static_assert(constant_coefficient_v<decltype(cp2_int.shiftLeft<1>())> == 4);
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftLeft<1>()}(), cdp2_int.shiftLeft<1>()(0, 0));
  EXPECT_EQ(0, cdp2_int.shiftLeft<1>()(0, 1));
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftLeft<1>()}(), 4);
  static_assert(constant_diagonal_matrix<decltype(cdp2_int.shiftLeft<1>()), CompileTimeStatus::unknown>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().cast<int>().array().shiftLeft<1>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cast<int>().array().shiftLeft<1>()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().cast<int>().shiftLeft<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cast<int>().array().shiftLeft<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().cast<int>().array().shiftLeft<1>())>);
#endif

  // scalar_real_op
  static_assert(constant_coefficient_v<decltype(real(C22_2 {std::declval<C22_2>()}))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(real(Cd22_2 {std::declval<Cd22_2>()}))> == 2);
  EXPECT_EQ((constant_coefficient{real(cxa)}()), 1);
  EXPECT_EQ((constant_coefficient{real(cxb)}()), 3);
  static_assert(hermitian_matrix<decltype(real(cxa))>);

  // scalar_imag_op
  static_assert(constant_coefficient_v<decltype(imag(C22_2 {std::declval<C22_2>()}))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(imag(Cd22_2 {std::declval<Cd22_2>()}))> == 0);
  EXPECT_EQ((constant_coefficient{imag(cxa)}()), 2);
  EXPECT_EQ((constant_coefficient{imag(cxb)}()), 4);
  static_assert(hermitian_matrix<decltype(imag(cxa))>);

  // scalar_real_ref_op -- Eigen::CwiseUnaryView
  static_assert(not self_contained<decltype(std::declval<CA22>().real())>);
  static_assert(not self_contained<decltype(std::declval<C22_2>().real())>);
  static_assert(constant_coefficient_v<decltype(M11::Identity().real())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<Z11>().real())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().real())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().real())> == 2);
  static_assert(constant_matrix<decltype(cxa.real())>);
  EXPECT_EQ(constant_coefficient{cxa.real()}(), 1);
  static_assert(constant_matrix<decltype(cxb.real())>);
  EXPECT_EQ(constant_coefficient{cxb.real()}(), 3);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().real())> == 2);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().real()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().real()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_m1>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Z22>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().real())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().real())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().real())>);
  static_assert(hermitian_matrix<decltype(cxa.real())>);

  // scalar_imag_ref_op -- Eigen::CwiseUnaryView
  static_assert(not self_contained<decltype(std::declval<CA22>().imag())>);
  static_assert(constant_coefficient_v<decltype(M11::Identity().imag())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z11>().imag())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().imag())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().imag())> == 0);
  static_assert(constant_matrix<decltype(cxa.imag())>);
  EXPECT_EQ(constant_coefficient{cxa.imag()}(), 2);
  static_assert(constant_matrix<decltype(cxb.imag())>);
  EXPECT_EQ(constant_coefficient{cxb.imag()}(), 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().imag())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().imag())> == 0);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().imag()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().imag()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().imag())>);
  static_assert(hermitian_matrix<decltype(cxa.imag())>);

  // scalar_exp_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().exp())>, internal::constexpr_exp(2)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().exp())>, internal::constexpr_exp(-2)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cp2.exp()}(), cp2.exp()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cm2.exp()}(), cm2.exp()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.exp()}(), cxa.exp()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.exp()}(), cxb.exp()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().exp())>);
  static_assert(not zero_matrix<decltype(zero.exp())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().exp()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * -3).exp())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().exp())>);
  static_assert(not hermitian_matrix<decltype(cxa.exp())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_expm1_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().expm1())>, internal::constexpr_expm1(2)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().expm1())>, internal::constexpr_expm1(-2)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cp2.expm1()}(), cp2.expm1()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cm2.expm1()}(), cm2.expm1()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.expm1()}(), cxa.expm1()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.expm1()}(), cxb.expm1()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().expm1())> == internal::constexpr_expm1(2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().expm1())> == internal::constexpr_expm1(-1));
  EXPECT_TRUE(are_within_tolerance<10>(constant_diagonal_coefficient{cdp2.expm1()}(), cdp2.expm1()(0, 0)));
  EXPECT_EQ(0, cdp2.expm1()(0, 1));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdm2.expm1()}(), cdm2.expm1()(0, 0)));
  EXPECT_EQ(0, cdm2.expm1()(0, 1));
  static_assert(zero_matrix<decltype(zero.expm1())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().expm1()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).expm1())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().expm1())>);
  static_assert(not hermitian_matrix<decltype(cxa.expm1())>); // because cxa is not necessarily hermitian
#endif

  // scalar_log_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log())>, internal::constexpr_log(2)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log()}(), cp2.log()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.log()}(), cxa.log()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxb.log()}(), cxb.log()(0, 0)));
  static_assert(zero_matrix<decltype(std::declval<C22_1>().log())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().log())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().log()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).log())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log())>);
  static_assert(not hermitian_matrix<decltype(cxa.log())>); // because cxa is not necessarily hermitian

  // scalar_log1p_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log1p())>, internal::constexpr_log1p(2)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log1p()}(), cp2.log1p()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.log1p()}(), cxa.log1p()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.log1p()}(), cxb.log1p()(0, 0)));
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().log1p())>, internal::constexpr_log1p(2)));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.log1p()}(), cdp2.log1p()(0, 0)));
  EXPECT_EQ(0, cdp2.log1p()(0, 1));
  static_assert(zero_matrix<decltype(zero.log1p())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().log1p()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).log1p())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log1p())>);
  static_assert(not hermitian_matrix<decltype(cxa.log1p())>); // because cxa is not necessarily hermitian

  // scalar_log10_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log10())>, internal::constexpr_log(2) / numbers::ln10_v<double>));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log10()}(), cp2.log10()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.log10()}(), cxa.log10()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.log10()}(), cxb.log10()(0, 0)));
  static_assert(zero_matrix<decltype(std::declval<C22_1>().log10())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().log10())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().log10()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).log10())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log10())>);
  static_assert(not hermitian_matrix<decltype(cxa.log10())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_log2_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log2())>, internal::constexpr_log(2) / numbers::ln2_v<double>));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log2()}(), cp2.log2()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.log2()}(), cxa.log2()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxb.log2()}(), cxb.log2()(0, 0)));
  static_assert(zero_matrix<decltype(std::declval<C22_1>().log2())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().log2())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().log2()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).log2())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log2())>);
  static_assert(not hermitian_matrix<decltype(cxa.log2())>); // because cxa is not necessarily hermitian
#endif

  // scalar_sqrt_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().sqrt())>, internal::constexpr_sqrt(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.sqrt()}(), cp2.sqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.sqrt()}(), cxa.sqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.sqrt()}(), cxb.sqrt()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().sqrt())> == internal::constexpr_sqrt(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.sqrt()}(), cdp2.sqrt()(0, 0)));
  EXPECT_EQ(0, cdp2.sqrt()(0, 1));
  static_assert(zero_matrix<decltype(zero.sqrt())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().sqrt()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).sqrt())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().sqrt())>);
  static_assert(not hermitian_matrix<decltype(cxa.sqrt())>); // because cxa is not necessarily hermitian

  // scalar_rsqrt_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().rsqrt())>, 1./internal::constexpr_sqrt(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.rsqrt()}(), cp2.rsqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.rsqrt()}(), cxa.rsqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.rsqrt()}(), cxb.rsqrt()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().rsqrt())>);
  static_assert(not zero_matrix<decltype(zero.rsqrt())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().rsqrt()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).rsqrt())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().rsqrt())>);
  static_assert(not hermitian_matrix<decltype(cxa.rsqrt())>); // because cxa is not necessarily hermitian

  // scalar_cos_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().cos())>, internal::constexpr_cos(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cos()}(), cp2.cos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cos()}(), cxa.cos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.cos()}(), cxb.cos()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().cos())>);
  static_assert(not zero_matrix<decltype(zero.cos())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().cos()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).cos())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().cos())>);
  static_assert(not hermitian_matrix<decltype(cxa.cos())>); // because cxa is not necessarily hermitian

  // scalar_sin_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().sin())>, internal::constexpr_sin(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.sin()}(), cp2.sin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.sin()}(), cxa.sin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.sin()}(), cxb.sin()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().sin())> == internal::constexpr_sin(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.sin()}(), cdp2.sin()(0, 0)));
  EXPECT_EQ(0, cdp2.sin()(0, 1));
  static_assert(zero_matrix<decltype(zero.sin())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().sin()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).sin())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().sin())>);
  static_assert(not hermitian_matrix<decltype(cxa.sin())>); // because cxa is not necessarily hermitian

  // scalar_tan_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().tan())>, internal::constexpr_tan(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.tan()}(), cp2.tan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.tan()}(), cxa.tan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.tan()}(), cxb.tan()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().tan())> == internal::constexpr_tan(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.tan()}(), cdp2.tan()(0, 0)));
  EXPECT_EQ(0, cdp2.tan()(0, 1));
  static_assert(zero_matrix<decltype(zero.tan())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().tan()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).tan())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().tan())>);
  static_assert(not hermitian_matrix<decltype(cxa.tan())>); // because cxa is not necessarily hermitian

  // scalar_acos_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_1>().acos())>, internal::constexpr_acos(1.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{(cp2/4).acos()}(), (cp2/4).acos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.acos()}(), cxa.acos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxb.acos()}(), cxb.acos()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_m1>().acos())>);
  static_assert(not zero_matrix<decltype(zero.acos())>);
  static_assert(not triangular_matrix<decltype((std::declval<Tuv22>()* 0.25).array().acos()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id* 0.25).acos())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>()* 0.25).array().acos())>);
  static_assert(not hermitian_matrix<decltype((cxa* 0.25).acos())>); // because cxa is not necessarily hermitian

  // scalar_asin_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_1>().asin())>, internal::constexpr_asin(1.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{(cp2/4).asin()}(), (cp2/4).asin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.asin()}(), cxa.asin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxb.asin()}(), cxb.asin()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().asin())> == internal::constexpr_asin(-1.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{(cdp2*0.25).asin()}(), (cdp2*0.25).asin()(0, 0)));
  EXPECT_EQ(0, (cdp2*0.25).asin()(0, 1));
  static_assert(zero_matrix<decltype(zero.asin())>);
  static_assert(triangular_matrix<decltype((std::declval<Tuv22>() * 0.25).array().asin()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 0.25).asin())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>() * 0.25).array().asin())>);
  static_assert(not hermitian_matrix<decltype((cxa* 0.25).asin())>); // because cxa is not necessarily hermitian

  // scalar_atan_op
  //static_assert(constexpr_atan(11.) > 1.1);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().atan())>, internal::constexpr_atan(2.)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cp2.atan()}(), cp2.atan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.atan()}(), cxa.atan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.atan()}(), cxb.atan()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m2>().atan())> == internal::constexpr_atan(-2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{(cdp2*0.25).atan()}(), (cdp2*0.25).atan()(0, 0)));
  EXPECT_EQ(0, (cdp2*0.25).atan()(0, 1));
  static_assert(zero_matrix<decltype(zero.atan())>);
  static_assert(triangular_matrix<decltype((std::declval<Tuv22>() * 0.25).array().atan()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 0.25).atan())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>() * 0.25).array().atan())>);
  static_assert(not hermitian_matrix<decltype((cxa * 0.25).atan())>); // because cxa is not necessarily hermitian

  // scalar_tanh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().tanh())>, internal::constexpr_tanh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.tanh()}(), cp2.tanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.tanh()}(), cxa.tanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.tanh()}(), cxb.tanh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().tanh())> == internal::constexpr_tanh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.tanh()}(), cdp2.tanh()(0, 0)));
  EXPECT_EQ(0, cdp2.tanh()(0, 1));
  static_assert(zero_matrix<decltype(zero.tanh())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().tanh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).tanh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().tanh())>);
  static_assert(not hermitian_matrix<decltype(cxa.tanh())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_atanh_op
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{(cp2 * 0.3).atanh()}(), (cp2 * 0.3).atanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.atanh()}(), cxa.atanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.atanh()}(), cxb.atanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{(cdp2 * 0.3).atanh()}(), (cdp2 * 0.3).atanh()(0, 0)));
  EXPECT_EQ(0, (cdp2 * 0.3).atanh()(0, 1));
  static_assert(zero_matrix<decltype(zero.atanh())>);
  static_assert(triangular_matrix<decltype((std::declval<Tuv22>() * 0.3).array().atanh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -0.9).atanh())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>() * 0.3).array().atanh())>);
  static_assert(not hermitian_matrix<decltype(cxa.atanh())>); // because cxa is not necessarily hermitian
#endif

  // scalar_sinh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().sinh())>, internal::constexpr_sinh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.sinh()}(), cp2.sinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.sinh()}(), cxa.sinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.sinh()}(), cxb.sinh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().sinh())> == internal::constexpr_sinh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.sinh()}(), cdp2.sinh()(0, 0)));
  EXPECT_EQ(0, cdp2.sinh()(0, 1));
  static_assert(zero_matrix<decltype(zero.sinh())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().sinh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).sinh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().sinh())>);
  static_assert(not hermitian_matrix<decltype(cxa.sinh())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_asinh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().asinh())>, internal::constexpr_asinh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.asinh()}(), cp2.asinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.asinh()}(), cxa.asinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.asinh()}(), cxb.asinh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().asinh())> == internal::constexpr_asinh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.asinh()}(), cdp2.asinh()(0, 0)));
  EXPECT_EQ(0, cdp2.asinh()(0, 1));
  static_assert(zero_matrix<decltype(zero.asinh())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().asinh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).asinh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().asinh())>);
  static_assert(not hermitian_matrix<decltype(cxa.asinh())>); // because cxa is not necessarily hermitian
#endif

  // scalar_cosh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().cosh())>, internal::constexpr_cosh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cosh()}(), cp2.cosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cosh()}(), cxa.cosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<100>(constant_coefficient{cxb.cosh()}(), cxb.cosh()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().cosh())>);
  static_assert(not zero_matrix<decltype(zero.cosh())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().cosh()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).cosh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().cosh())>);
  static_assert(not hermitian_matrix<decltype(cxa.cosh())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_acosh_op
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cp2.acosh()}(), cp2.acosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxa.acosh()}(), cxa.acosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance<10>(constant_coefficient{cxb.acosh()}(), cxb.acosh()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().acosh())>);
  static_assert(not zero_matrix<decltype(zero.acosh())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().acosh()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).acosh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().acosh())>);
  static_assert(not hermitian_matrix<decltype(cxa.acosh())>); // because cxa is not necessarily hermitian
#endif

  // scalar_inverse_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().inverse())>, 0.5));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.inverse()}(), cp2.inverse()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.inverse()}(), cxa.inverse()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().inverse())>);
  static_assert(not zero_matrix<decltype(zero.inverse())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().inverse()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).inverse())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().inverse())>);
  static_assert(not hermitian_matrix<decltype(cxa.inverse())>); // because cxa is not necessarily hermitian

  // scalar_square_op
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().square())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().square())> == 4);
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.square()}(), cp2.square()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.square()}(), cxa.square()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.square()}(), cxb.square()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().square())> == 4);
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.square()}(), cdp2.square()(0, 0)));
  EXPECT_EQ(0, cdp2.square()(0, 1));
  static_assert(zero_matrix<decltype(zero.square())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().square()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).square())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().square())>);
  static_assert(not hermitian_matrix<decltype(cxa.square())>); // because cxa is not necessarily hermitian

  // scalar_cube_op
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().cube())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().cube())> == -8);
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cube()}(), cp2.cube()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cube()}(), cxa.cube()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().cube())> == 8);
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.cube()}(), cdp2.cube()(0, 0)));
  EXPECT_EQ(0, cdp2.cube()(0, 1));
  static_assert(zero_matrix<decltype(zero.cube())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().cube()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).cube())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().cube())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().cube())>);
  static_assert(not hermitian_matrix<decltype(cxa.cube())>); // because cxa is not necessarily hermitian

  // Eigen::internal::scalar_round_op not implemented
  // Eigen::internal::scalar_floor_op not implemented
  // Eigen::internal::scalar_rint_op not implemented (Eigen 3.4+)
  // Eigen::internal::scalar_ceil_op not implemented

  // Eigen::internal::scalar_isnan_op not implemented
  // Eigen::internal::scalar_isinf_op not implemented
  // Eigen::internal::scalar_isfinite_op not implemented

  // scalar_boolean_not_op
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_true>())> == false);
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_false>())> == true);
  static_assert(constant_coefficient_v<decltype(not std::declval<C22_1>())> == false); // requires narrowing from 1 to true.
  static_assert(constant_coefficient_v<decltype(not std::declval<Z22>())> == true); // requires narrowing from 0 to false.
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_true>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_false>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<C22_1>())> == false); // requires narrowing from 1 to true.
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<Z11>())> == true); // requires narrowing from 0 to false.
  static_assert(zero_matrix<decltype(not std::declval<B22_true>())>);
  static_assert(not diagonal_matrix<decltype(not std::declval<B22_false>())>);
  static_assert(hermitian_matrix<decltype(not std::declval<B22_false>())>);

  // Eigen::internal::scalar_sign_op not implemented

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_logistic_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().logistic())>, 1 / (1 + (internal::constexpr_exp(-2)))));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().logistic())>, 1 / (1 + (internal::constexpr_exp(2)))));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.logistic()}(), cp2.logistic()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.logistic()}(), cxa.logistic()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.logistic()}(), cxb.logistic()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().logistic())>);
  static_assert(not zero_matrix<decltype(zero.logistic())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().logistic()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).logistic())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().logistic())>);
  static_assert(not hermitian_matrix<decltype(cxa.logistic())>); // because cxa is not necessarily hermitian
#endif

  // bind1st_op
  using CB1sum = Eigen::internal::bind1st_op<Eigen::internal::scalar_sum_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>{cp2, CB1sum{3}}}()), 5);
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cm2)>{cm2, CB1sum{6}}}()), 4);
  static_assert(constant_matrix<Eigen::CwiseUnaryOp<CB1sum, Z22>, CompileTimeStatus::unknown>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB1sum, Z22>>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cdp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>, Likelihood::maybe>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cm2)>>);

  using CB1prod = Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1prod, decltype(cm2)>{cm2, CB1prod{6}}}()), -12);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB1prod, decltype(cdp2)>{cdp2, CB1prod{3}}}()), 6);
  static_assert(constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, I22>, CompileTimeStatus::unknown>);
  static_assert(zero_matrix<Eigen::CwiseUnaryOp<CB1prod, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, I22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, DM2>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cp2)>, Likelihood::maybe>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cm2)>>);

  // bind2nd_op
  using CB2sum = Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>{cp2, CB2sum{3}}}()), 5);
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cm2)>{cm2, CB2sum{6}}}()), 4);
  static_assert(constant_matrix<Eigen::CwiseUnaryOp<CB2sum, Z22>, CompileTimeStatus::unknown>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB2sum, Z22>>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cdp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>, Likelihood::maybe>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cm2)>>);

  using CB2prod = Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>{cp2, CB2prod{3}}}()), 6);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB2prod, decltype(cdm2)>{cdm2, CB2prod{6}}}()), -12);
  static_assert(constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, I22>, CompileTimeStatus::unknown>);
  static_assert(zero_matrix<Eigen::CwiseUnaryOp<CB2prod, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, I22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, DM2>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>, Likelihood::maybe>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cm2)>>);
}


TEST(eigen3, cwise_binary_operations)
{
  auto id = I22 {2, 2}; // Identity
  auto cid = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<cdouble>, CA22> {2, 2};
  auto zero = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2

  // general CwiseBinaryOp
  static_assert(self_contained<decltype(2 * std::declval<I22>() + std::declval<I22>())>);
  static_assert(not self_contained<decltype(2 * std::declval<I22>() + A22 {1, 2, 3, 4})>);
  static_assert(row_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I22>() + std::declval<I22>())>> == 2);
  static_assert(column_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I22>() + std::declval<I22>())>> == 2);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, A22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const A22, const A22>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const A22,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>>>);

  // scalar_sum_op
  static_assert(index_dimension_of_v<decltype(std::declval<M2x>() + std::declval<Mx3>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(std::declval<Mx3>() + std::declval<M2x>()), 1> == 3);

  static_assert(one_by_one_matrix<decltype(std::declval<M11>() + std::declval<Mx1>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<M1x>() + std::declval<M11>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<M1x>() + std::declval<Mx1>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<I11>() + std::declval<Axx>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<DMx>() + std::declval<M1x>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<DMx>() + std::declval<Mx1>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<Axx>() + std::declval<A11>())>);
  static_assert(one_by_one_matrix<decltype(std::declval<Mxx>() + std::declval<Mx1>()), Likelihood::maybe>);
  static_assert(not one_by_one_matrix<decltype(std::declval<Mxx>() + std::declval<Mx1>())>);

  static_assert(square_matrix<decltype(std::declval<M2x>() + std::declval<Mx2>())>);
  static_assert(not square_matrix<decltype(std::declval<Mxx>() + std::declval<Mx2>())>);
  static_assert(square_matrix<decltype(std::declval<Mxx>() + std::declval<Mx2>()), Likelihood::maybe>);
  static_assert(square_matrix<decltype(std::declval<DMx>() + std::declval<DMx>())>);
  static_assert(square_matrix<decltype(std::declval<DMx>() + std::declval<Mxx>())>);

  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_3>())> == 5);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_m2>())> == 0);
  EXPECT_EQ((constant_coefficient{cxa + cxb}()), (std::complex<double>{4, 6}));
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() + M22::Constant(3).array()}(), 5);
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() + zero}(), 2);
  EXPECT_EQ(constant_coefficient {zero + M22::Constant(3).array()}(), 3);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() + std::declval<Cd22_3>())> == 5);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() + std::declval<C22_m2>())> == 0);
  EXPECT_EQ(constant_diagonal_coefficient {id + id}(), 2);
  EXPECT_EQ((constant_diagonal_coefficient {I11{1, 1} + M11::Constant(3).array()}()), 4);

  static_assert(zero_matrix<decltype(std::declval<C22_2>() + std::declval<C22_m2>())>);
  static_assert(zero_matrix<decltype(std::declval<C21_2>() + std::declval<C21_m2>())>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>() + std::declval<DW21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW2x>() + std::declval<DW2x>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DWx1>() + std::declval<DWx1>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DWxx>() + std::declval<DWxx>())>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>() + std::declval<Tlv22>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>() + std::declval<Tuv22>()), TriangleType::upper>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>() + std::declval<Tlv22>())>);

  static_assert(hermitian_matrix<decltype(std::declval<DW21>() + std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() + std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() + std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() + std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() + std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() + std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv2x>() + std::declval<Sauvx2>())>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>() + std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() + std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() + std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() + std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() + std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv2x>() + std::declval<Sauvx2>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() + std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv2x>() + std::declval<Sauvx2>())>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>() + std::declval<Salv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>() + std::declval<Sauv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>() + std::declval<Salv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>() + std::declval<Sauv22>())> == HermitianAdapterType::any);

  static_assert(not writable<decltype(std::declval<M22>() + std::declval<M22>())>);
  static_assert(not modifiable<decltype(std::declval<M33>() + std::declval<M33>()), M33>);


  // scalar_product_op
  static_assert(index_dimension_of_v<decltype(std::declval<A2x>() * std::declval<Ax2>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(std::declval<A2x>() * std::declval<Ax2>()), 1> == 2);

  static_assert(not constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx::ConstantReturnType, Mxx>>);
  static_assert(constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx::ConstantReturnType, Mxx>, CompileTimeStatus::unknown, Likelihood::maybe>);

  static_assert(not constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx, Mxx::ConstantReturnType>>);
  static_assert(constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx, Mxx::ConstantReturnType>, CompileTimeStatus::unknown, Likelihood::maybe>);

  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() * std::declval<C21_m2>())> == -4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() * std::declval<Z21>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>() * std::declval<C21_2>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<M21>().array() * std::declval<Z21>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>() * std::declval<M21>().array())> == 0);
  EXPECT_EQ((constant_coefficient{cxa * cxb}()), (std::complex<double>{-5, 10}));
  EXPECT_EQ((constant_coefficient{cp2 * cm2}()), -4);
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() * M22::Constant(3).array()}(), 6);
  EXPECT_EQ((constant_coefficient{zero * M22::Constant(3).array()}()), 0);
  EXPECT_EQ((constant_coefficient{M22::Constant(2).array() * zero}()), 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<Cd22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<A22>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<A22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() * std::declval<Cd22_3>())> == 6); // no conjugate-product test
  static_assert(constant_diagonal_matrix<decltype(cdp2), CompileTimeStatus::unknown>);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2}()), 2);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * cdm2}()), -4);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * zero}()), 0);
  EXPECT_EQ((constant_diagonal_coefficient{zero * cdm2}()), 0);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * id}()), 2);
  EXPECT_EQ((constant_diagonal_coefficient{id * cdm2}()), -2);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>() * std::declval<DW21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>().array() * std::declval<M22>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>() * 2)>);
  static_assert(diagonal_matrix<decltype(3 * std::declval<DW21>())>);
  static_assert(not diagonal_matrix<decltype(std::declval<DW21>().array() / std::declval<DW21>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tlv22>().array() * std::declval<Tuv22>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tuv22>().array() * std::declval<Tlv22>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tuv2x>().array() * std::declval<Tlvx2>().array())>);
  static_assert(not diagonal_matrix<decltype(std::declval<Tuvxx>().array() * std::declval<Tlvxx>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tuvxx>().array() * std::declval<Tlvxx>().array()), Likelihood::maybe>);

  static_assert(triangular_matrix<decltype(std::declval<DW21>() * 3), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(2 * std::declval<DW21>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array() * std::declval<Tlv22>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv2x>().array() * std::declval<Tlvx2>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlvxx>().array() * std::declval<Tlvxx>().array()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<DW21>() * 3), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(2 * std::declval<DW21>()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array() * std::declval<Tuv22>().array()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv2x>().array() * std::declval<Tuvx2>().array()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuvxx>().array() * std::declval<Tuvxx>().array()), TriangleType::upper, Likelihood::maybe>);

  static_assert(hermitian_matrix<decltype(std::declval<DW21>().array() * std::declval<DW21>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>() * 3)>);
  static_assert(hermitian_matrix<decltype(2 * std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array() * std::declval<Salv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() * std::declval<Salv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array() * std::declval<Sauv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>() * 3)>);
  static_assert(hermitian_matrix<decltype(2 * std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() * std::declval<Sauv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() * std::declval<Salv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array() * std::declval<Sauv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() * std::declval<Salv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array() * std::declval<Sauv22>().array())>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>().array() * std::declval<Sauv22>().array())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>().array() * std::declval<Salv22>().array())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>().array() * std::declval<Salv22>().array())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>().array() * std::declval<Sauv22>().array())> == HermitianAdapterType::any);

  static_assert(not writable<decltype(std::declval<I22>() * 2)>);
  static_assert(not modifiable<decltype(M33::Identity() * 2), decltype(M33::Identity() * 2)>);

  // scalar_conj_product_op
  using CProd = Eigen::internal::scalar_conj_product_op<std::complex<double>, std::complex<double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseBinaryOp<CProd, decltype(cxa), decltype(cxb)>{cxa, cxb}}()), (std::complex<double>{11, -2}));
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseBinaryOp<CProd, decltype(cxa), decltype(cid)>{cxa, cid}}()), (std::complex<double>{1, -2}));
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseBinaryOp<CProd, decltype(cid), decltype(cxb)>{cid, cxb}}()), (std::complex<double>{3, 4}));
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<CProd, DW21, DW21>>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<CProd, Tlv22, Tlv22>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<CProd, Tlv22, M22>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<CProd, Tuv22, Tuv22>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<CProd, M22, Tuv22>, TriangleType::upper>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<CProd, Sauv22, Salv22>>);
  static_assert(OpenKalman::interface::IndexibleObjectTraits<Eigen::CwiseBinaryOp<CProd, Sauv22, DW21>>::is_hermitian == true);
  static_assert(OpenKalman::interface::IndexibleObjectTraits<Eigen::CwiseBinaryOp<CProd, DW21, DW21>>::is_hermitian == true);

  // scalar_min_op
  static_assert(constant_coefficient_v<decltype(cp2.min(cm2))> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().min(std::declval<C21_m2>()))> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().min(std::declval<Cd22_3>()))> == 2);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>().array().min(std::declval<DW21>().array()))>);
  static_assert(not diagonal_matrix<decltype(std::declval<DW21>().array().min(std::declval<M21>().array()))>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array().min(std::declval<Tlv22>().array())), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().min(std::declval<Tuv22>().array())), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().min(std::declval<Salv22>().array()))>);

  // scalar_max_op
  static_assert(constant_coefficient_v<decltype(cp2.max(cm2))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().max(std::declval<C21_m2>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().max(std::declval<Cd22_3>()))> == 3);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>().array().max(std::declval<DW21>().array()))>);
  static_assert(not diagonal_matrix<decltype(std::declval<DW21>().array().max(std::declval<M21>().array()))>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array().max(std::declval<Tlv22>().array())), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().max(std::declval<Tuv22>().array())), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().max(std::declval<Salv22>().array()))>);

  // scalar_cmp_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_EQ>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, false));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LT>, decltype(cm2), decltype(cp2)>{cm2, cp2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LE>, decltype(cp2), decltype(cp2)>{cp2, cp2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_GT>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_GE>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_NEQ>, decltype(cm2), decltype(cm2)>{cm2, cm2})>, false));
  // No test for Eigen::internal::ComparisonName::cmp_UNORD
  static_assert(not diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_EQ>, M22, M22>>);
  static_assert(not triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LT>, M22, M22>>);

  // scalar_hypot_op
  using CWHYP = Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, decltype(cp2), decltype(cm2)>;
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(CWHYP{cp2, cm2})>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, C21_2, C21_m2>>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Cd22_2, Cd22_3>>, OpenKalman::internal::constexpr_sqrt(13.)));
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, DW21, DW21>>);
  static_assert(not diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, M21, DW21>>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Tlv22, Tlv22>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Tuv22, Tuv22>, TriangleType::upper>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Sauv22, Salv22>>);

  // scalar_pow_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(cp2.pow(cm2))>, 0.25));
  using M11_int = eigen_matrix_t<int, 1, 1>;
  using C11_3_int = decltype(M11_int::Identity() + M11_int::Identity() + M11_int::Identity());
  using C21_3_int = Eigen::Replicate<C11_3_int, 2, 1>;
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_3_int>().array().pow(std::declval<C21_3_int>().array()))>, 27));
  static_assert(not diagonal_matrix<decltype(std::declval<DW21>().array().pow(std::declval<DW21>().array()))>);
  static_assert(not triangular_matrix<decltype(std::declval<Sauv22>().array().pow(std::declval<Sauv22>().array()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().pow(std::declval<Salv22>().array()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().pow(std::declval<Sauv22>().array()))>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>().array().pow(std::declval<Salv22>().array()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().pow(std::declval<DW21>().array()))>);

  // scalar_difference_op
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() - std::declval<C21_2>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() - std::declval<C21_m2>())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() - std::declval<C22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() - std::declval<Cd22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() - std::declval<Cd22_3>())> == -1);
  static_assert(zero_matrix<decltype(std::declval<C22_2>() - std::declval<C22_2>())>);
  static_assert(zero_matrix<decltype(std::declval<C21_2>() - std::declval<C21_2>())>);
  static_assert(identity_matrix<decltype(M33::Identity() - (M33::Identity() - M33::Identity()))>);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>() - std::declval<DW21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW2x>() - std::declval<DWx1>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DWx1>() - std::declval<DW21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DWxx>() - std::declval<DWxx>())>);
  static_assert(triangular_matrix<decltype(std::declval<DW21>() - std::declval<DW21>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<DW21>() - std::declval<DW21>()), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>() - std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() - std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salvxx>() - std::declval<Sauvxx>()), Likelihood::maybe>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>() - std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauvxx>() - std::declval<Sauvxx>()), Likelihood::maybe>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salvxx>() - std::declval<Sauvxx>()), Likelihood::maybe>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>() - std::declval<Salv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>() - std::declval<Sauv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>() - std::declval<Salv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>() - std::declval<Sauv22>())> == HermitianAdapterType::any);
  EXPECT_EQ((constant_coefficient{cxa - cxb}()), (std::complex<double>{-2, -2}));

  // scalar_quotient_op
  static_assert(constant_coefficient_v<decltype(std::declval<C11_3>() / std::declval<C11_m2>())> == -1.5);
  static_assert(constant_coefficient_v<decltype(std::declval<Z11>() / std::declval<C11_3>())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C11_3>() / std::declval<Z11>()), CompileTimeStatus::known>); // divide by zero
  static_assert(constant_matrix<decltype(std::declval<C11_3>() / std::declval<Z11>()), CompileTimeStatus::unknown>); // divide by zero, but determined at runtime
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() / std::declval<C21_m2>())> == -1);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>() / std::declval<C21_m2>())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C21_2>() / std::declval<Z21>())>); // divide by zero
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_3>() / std::declval<C11_m2>())> == -1.5);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z11>() / std::declval<C11_m2>())> == 0);
  static_assert(constant_diagonal_matrix<decltype(std::declval<C11_3>().array() / std::declval<Z11>().array()), CompileTimeStatus::unknown>); // divide by zero
  static_assert(not diagonal_matrix<decltype(std::declval<DW21>().array() / std::declval<DW21>().array())>);
  static_assert(not triangular_matrix<decltype(std::declval<DW21>().array() / std::declval<DW21>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>().array() / std::declval<DW21>().array())>);
  using CWQ = Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<std::complex<double>, std::complex<double>>, decltype(cxa), decltype(cxb)>;
  EXPECT_EQ((constant_coefficient{CWQ{cxa, cxb}}()), (std::complex<double>{11./25, 2./25}));

  // scalar_boolean_and_op
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>() and std::declval<B22_true>())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>() and std::declval<B22_false>())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>() and std::declval<B22_false>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() and std::declval<BI22>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() and std::declval<B22_false>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() and std::declval<B22_true>())> == true);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>() and std::declval<DW21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>().array() and std::declval<B22_true>())>);
  static_assert(diagonal_matrix<decltype(std::declval<B22_true>() and std::declval<DW21>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tlv22>().array() and std::declval<Tuv22>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tuv22>().array() and std::declval<Tlv22>().array())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array() and std::declval<Tlv22>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array() and std::declval<A22>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<A22>() and std::declval<Tlv22>().array()), TriangleType::lower>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() and std::declval<Salv22>().array())>);

  // scalar_boolean_or_op
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>() or std::declval<B22_true>())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>() or std::declval<B22_false>())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>() or std::declval<B22_true>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() or std::declval<BI22>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() or std::declval<B22_false>())> == true);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>() or std::declval<DW21>())>);
  static_assert(not (diagonal_matrix<decltype(std::declval<DW21>().array() or std::declval<M21>().array())>));
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array() or std::declval<Tlv22>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array() or std::declval<Tuv22>().array()), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() or std::declval<Salv22>().array())>);

  // scalar_boolean_xor_op
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>() xor std::declval<B22_true>())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>() xor std::declval<B22_true>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() xor std::declval<BI22>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>() xor std::declval<BI22>())> == true);
  static_assert(not diagonal_matrix<decltype(std::declval<DW21>().array() xor std::declval<DW21>().array())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array() xor std::declval<Tuv22>().array())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array() xor std::declval<Salv22>().array())>);

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_absolute_difference_op
  using AD = Eigen::internal::scalar_absolute_difference_op<double, double>;
  static_assert(constant_coefficient_v<Eigen::CwiseBinaryOp<AD, C21_2, C21_m2>> == 4);
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<AD, DW21, DW21>>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<AD, Tlv22, Tlv22>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<AD, Tuv22, Tuv22>, TriangleType::upper>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<AD, Sauv22, Salv22>>);
#endif

}


TEST(eigen3, cwise_ternary_operations)
{
  // No current tests for cwise ternary operations
}
