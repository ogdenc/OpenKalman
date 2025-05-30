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
using namespace OpenKalman::test;

namespace
{
  auto id = I22 {2, 2}; // Identity
  auto z = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2
}

/// \todo Add tests for stl operators

TEST(eigen3, Eigen_CwiseUnaryOp_scalar_opposite_op)
{
  static_assert(constant_coefficient_v<decltype(-std::declval<C11_2>())> == -2);
  static_assert(constant_coefficient_v<decltype(-std::declval<C22_2>())> == -2);
  static_assert(constant_coefficient_v<decltype(-cp2)> == -2);
  static_assert(constant_coefficient_v<decltype(-cm2)> == 2);
  EXPECT_EQ((constant_coefficient{-cp2}()), -2);
  static_assert(constant_coefficient_v<decltype(-(-std::declval<C11_2>()))> == 2);

  static_assert(not constant_matrix<decltype(-std::declval<Cd22_m1>())>);
  static_assert(not constant_matrix<decltype(-std::declval<Ixx>())>);
  static_assert(constant_matrix<decltype(-std::declval<Cxx_m1>())>);
  static_assert(not constant_matrix<decltype(-std::declval<Cdxx_m1>())>);

  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<C11_2>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<I22>())> == -1);
  EXPECT_EQ((constant_diagonal_coefficient{-cdm2}()), 2);
  static_assert(constant_diagonal_coefficient_v<decltype(-(-std::declval<C11_m2>()))> == -2);

  static_assert(zero<decltype(-z)>);
  static_assert(identity_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(triangular_matrix<decltype(-std::declval<Tlv22>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(-std::declval<Tuv22>()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(-z)>);
  static_assert(diagonal_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<I22>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<I2x>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<Ix2>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<Ixx>())>);
  static_assert(hermitian_matrix<decltype(-std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(-std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(-z)>);
  static_assert(not hermitian_matrix<decltype(cxa)>);
  static_assert(not hermitian_matrix<decltype(-cxb)>);
  static_assert(not writable<decltype(-std::declval<M22>())>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_abs_op)
{
  using Ident11 = const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, Eigen::Array<double, 1, 1>>;
  static_assert(constant_coefficient_v<Ident11> == 1);
  using C211 = const Eigen::Replicate<const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<double>, Ident11, Ident11>, 2, 2>;
  static_assert(constant_coefficient_v<C211> == 2);
  using Traits = Eigen3::UnaryFunctorTraits<Eigen::internal::scalar_abs_op<double>>;
  static_assert(Traits::constexpr_operation()(-2) == 2);
  static_assert(Eigen3::constexpr_unary_operation_defined<Eigen::internal::scalar_abs_op<double>>);
  auto c22 = constant_coefficient {cp2.abs().nestedExpression()};
  static_assert(std::decay_t<decltype(c22)>::value == 2);
  static_assert(OpenKalman::values::operation {Traits::constexpr_operation(), c22} == 2);
  static_assert(constant_coefficient_v<const Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<double>, C211>> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().abs())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().abs())> == 2);
  static_assert(constant_coefficient_v<decltype(-std::declval<C22_m2>().abs())> == -2);
  static_assert(constant_coefficient_v<decltype((-std::declval<C22_m2>()).abs())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().abs())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().abs())> == 1);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<Cd22_2>().abs())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype((-std::declval<Cd22_2>()).abs())> == 2);
  static_assert(not constant_matrix<decltype(std::declval<Cd22_m1>().abs())>);
  static_assert(constant_matrix<decltype(std::declval<Cxx_m1>().abs())>);
  static_assert(not constant_matrix<decltype(std::declval<Cdxx_m1>().abs())>);
  static_assert(identity_matrix<decltype((-id).abs())>);
  static_assert(identity_matrix<decltype(std::declval<C11_m1>().abs())>);
  static_assert(zero<decltype(z.abs())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cwiseAbs()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).abs())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cwiseAbs())>);
  static_assert(hermitian_matrix<decltype(cxa.abs())>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_score_coeff_op)
{
  // scalar_score_coeff_op inherits from scalar_abs_op
  static_assert(constant_diagonal_coefficient_v<Eigen::CwiseUnaryOp<Eigen::internal::scalar_score_coeff_op<double>, Cd22_m2>> == 2);

}


// abs_knowing_score not implemented because it is not a true Eigen functor.


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_abs2_op)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().abs2())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<Cxx_m2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cdxx_2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().abs2())> == 1);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cdxx_m1>().abs2())> == 1);
  static_assert(not constant_matrix<decltype(std::declval<Cd22_m1>().abs2())>);
  static_assert(constant_matrix<decltype(std::declval<Cxx_m1>().abs2())>);
  static_assert(not constant_matrix<decltype(std::declval<Cdxx_m1>().abs2())>);
  static_assert(identity_matrix<decltype((-id).abs2())>);
  static_assert(identity_matrix<decltype(std::declval<C11_m1>().abs2())>);
  static_assert(zero<decltype(z.abs2())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cwiseAbs2()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).abs2())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cwiseAbs2())>);
  static_assert(hermitian_matrix<decltype(cxa.abs2())>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_conjugate_op)
{
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
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_arg_op)
{
  EXPECT_EQ(constant_coefficient{cp2.arg()}(), std::arg(2));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().arg())> == 0);
  EXPECT_EQ(constant_coefficient{cm2.arg()}(), std::arg(-2));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().arg())> == OpenKalman::numbers::pi);
  EXPECT_TRUE(values::internal::near<10>((constant_coefficient{cxa.arg()}()), (std::arg(std::complex<double>{1, 2}))));
  EXPECT_TRUE(values::internal::near<10>((constant_coefficient{cxb.arg()}()), (std::arg(std::complex<double>{3, 4}))));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().arg())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().cwiseAbs()), TriangleType::lower>);
  static_assert(not diagonal_matrix<decltype(std::declval<Cd22_2>().arg())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cwiseAbs()), TriangleType::upper>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cwiseAbs())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().cwiseAbs())>);
  static_assert(hermitian_matrix<decltype(cxa.arg())>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_cast_op)
{
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

}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
namespace
{
  auto id1_int = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<int>, Eigen::Array<int, 1, 1>> {1, 1}; // Identity
  auto id2_int = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<int>, Eigen::Array<int, 2, 2>> {2, 2}; // Identity
  auto cp2_int = (id1_int + id1_int).replicate<2, 2>(); // Constant +2
  auto cdp2_int = id2_int * 2; // Constant diagonal +2
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_shift_right_op)
{
  EXPECT_EQ(constant_coefficient{cp2_int.shiftRight<1>()}(), cp2_int.shiftRight<1>()(0, 0));
  EXPECT_EQ(constant_coefficient{cp2_int.shiftRight<1>()}(), cp2_int.shiftRight<1>()(0, 1));
  static_assert(constant_coefficient_v<decltype(cp2_int.shiftRight<1>())> == 1);
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftRight<1>()}(), cdp2_int.shiftRight<1>()(0, 0));
  EXPECT_EQ(0, cdp2_int.shiftRight<1>()(0, 1));
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftRight<1>()}(), 1);
  static_assert(values::dynamic<constant_diagonal_coefficient<decltype(cdp2_int.shiftRight<1>())>>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().cast<int>().array().shiftRight<1>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cast<int>().array().shiftRight<1>()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().cast<int>().shiftRight<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cast<int>().array().shiftRight<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().cast<int>().array().shiftRight<1>())>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_shift_left_op)
{
  EXPECT_EQ(constant_coefficient{cp2_int.shiftLeft<1>()}(), cp2_int.shiftLeft<1>()(0, 0));
  EXPECT_EQ(constant_coefficient{cp2_int.shiftLeft<1>()}(), cp2_int.shiftLeft<1>()(0, 1));
  static_assert(constant_coefficient_v<decltype(cp2_int.shiftLeft<1>())> == 4);
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftLeft<1>()}(), cdp2_int.shiftLeft<1>()(0, 0));
  EXPECT_EQ(0, cdp2_int.shiftLeft<1>()(0, 1));
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftLeft<1>()}(), 4);
  static_assert(values::dynamic<constant_diagonal_coefficient<decltype(cdp2_int.shiftLeft<1>())>>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().cast<int>().array().shiftLeft<1>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().cast<int>().array().shiftLeft<1>()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().cast<int>().shiftLeft<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().cast<int>().array().shiftLeft<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().cast<int>().array().shiftLeft<1>())>);
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_real_op)
{
  static_assert(constant_coefficient_v<decltype(real(C22_2 {std::declval<C22_2>()}))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(real(Cd22_2 {std::declval<Cd22_2>()}))> == 2);
  EXPECT_EQ((constant_coefficient{real(cxa)}()), 1);
  EXPECT_EQ((constant_coefficient{real(cxb)}()), 3);
  static_assert(hermitian_matrix<decltype(real(cxa))>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_imag_op)
{
  static_assert(constant_coefficient_v<decltype(imag(C22_2 {std::declval<C22_2>()}))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(imag(Cd22_2 {std::declval<Cd22_2>()}))> == 0);
  EXPECT_EQ((constant_coefficient{imag(cxa)}()), 2);
  EXPECT_EQ((constant_coefficient{imag(cxb)}()), 4);
  static_assert(hermitian_matrix<decltype(imag(cxa))>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_exp_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().exp())>, values::exp(2)));
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_m2>().exp())>, values::exp(-2)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cp2.exp()}(), cp2.exp()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cm2.exp()}(), cm2.exp()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.exp()}(), cxa.exp()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.exp()}(), cxb.exp()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().exp())>);
  static_assert(not zero<decltype(z.exp())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().exp()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * -3).exp())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().exp())>);
  static_assert(not hermitian_matrix<decltype(cxa.exp())>); // because cxa is not necessarily hermitian
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_CwiseUnaryOp_scalar_expm1_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().expm1())>, values::expm1(2)));
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_m2>().expm1())>, values::expm1(-2)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cp2.expm1()}(), cp2.expm1()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cm2.expm1()}(), cm2.expm1()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.expm1()}(), cxa.expm1()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.expm1()}(), cxb.expm1()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().expm1())> == values::expm1(2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().expm1())> == values::expm1(-1));
  EXPECT_TRUE(values::internal::near<10>(constant_diagonal_coefficient{cdp2.expm1()}(), cdp2.expm1()(0, 0)));
  EXPECT_EQ(0, cdp2.expm1()(0, 1));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdm2.expm1()}(), cdm2.expm1()(0, 0)));
  EXPECT_EQ(0, cdm2.expm1()(0, 1));
  static_assert(zero<decltype(z.expm1())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().expm1()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).expm1())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().expm1())>);
  static_assert(not hermitian_matrix<decltype(cxa.expm1())>); // because cxa is not necessarily hermitian
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_log_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().log())>, values::log(2)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.log()}(), cp2.log()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.log()}(), cxa.log()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxb.log()}(), cxb.log()(0, 0)));
  static_assert(zero<decltype(std::declval<C22_1>().log())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().log())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().log()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).log())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log())>);
  static_assert(not hermitian_matrix<decltype(cxa.log())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_log1p_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().log1p())>, values::log1p(2)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.log1p()}(), cp2.log1p()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.log1p()}(), cxa.log1p()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.log1p()}(), cxb.log1p()(0, 0)));
  static_assert(values::internal::near(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().log1p())>, values::log1p(2)));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.log1p()}(), cdp2.log1p()(0, 0)));
  EXPECT_EQ(0, cdp2.log1p()(0, 1));
  static_assert(zero<decltype(z.log1p())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().log1p()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -3).log1p())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log1p())>);
  static_assert(not hermitian_matrix<decltype(cxa.log1p())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_log10_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().log10())>, values::log(2) / numbers::ln10_v<double>));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.log10()}(), cp2.log10()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.log10()}(), cxa.log10()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.log10()}(), cxb.log10()(0, 0)));
  static_assert(zero<decltype(std::declval<C22_1>().log10())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().log10())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().log10()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).log10())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log10())>);
  static_assert(not hermitian_matrix<decltype(cxa.log10())>); // because cxa is not necessarily hermitian
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_CwiseUnaryOp_scalar_log2_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().log2())>, values::log(2) / numbers::ln2_v<double>));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.log2()}(), cp2.log2()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.log2()}(), cxa.log2()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxb.log2()}(), cxb.log2()(0, 0)));
  static_assert(zero<decltype(std::declval<C22_1>().log2())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().log2())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().log2()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).log2())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().log2())>);
  static_assert(not hermitian_matrix<decltype(cxa.log2())>); // because cxa is not necessarily hermitian
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_sqrt_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().sqrt())>, values::sqrt(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.sqrt()}(), cp2.sqrt()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.sqrt()}(), cxa.sqrt()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.sqrt()}(), cxb.sqrt()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().sqrt())> == values::sqrt(2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.sqrt()}(), cdp2.sqrt()(0, 0)));
  EXPECT_EQ(0, cdp2.sqrt()(0, 1));
  static_assert(zero<decltype(z.sqrt())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().sqrt()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).sqrt())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().sqrt())>);
  static_assert(not hermitian_matrix<decltype(cxa.sqrt())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_rsqrt_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().rsqrt())>, 1./values::sqrt(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.rsqrt()}(), cp2.rsqrt()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.rsqrt()}(), cxa.rsqrt()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.rsqrt()}(), cxb.rsqrt()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().rsqrt())>);
  static_assert(not zero<decltype(z.rsqrt())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().rsqrt()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).rsqrt())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().rsqrt())>);
  static_assert(not hermitian_matrix<decltype(cxa.rsqrt())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_cos_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().cos())>, values::cos(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.cos()}(), cp2.cos()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.cos()}(), cxa.cos()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.cos()}(), cxb.cos()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().cos())>);
  static_assert(not zero<decltype(z.cos())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().cos()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).cos())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().cos())>);
  static_assert(not hermitian_matrix<decltype(cxa.cos())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_sin_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().sin())>, values::sin(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.sin()}(), cp2.sin()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxa.sin()}(), cxa.sin()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.sin()}(), cxb.sin()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().sin())> == values::sin(2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.sin()}(), cdp2.sin()(0, 0)));
  EXPECT_EQ(0, cdp2.sin()(0, 1));
  static_assert(zero<decltype(z.sin())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().sin()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).sin())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().sin())>);
  static_assert(not hermitian_matrix<decltype(cxa.sin())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_tan_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().tan())>, values::tan(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.tan()}(), cp2.tan()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.tan()}(), cxa.tan()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.tan()}(), cxb.tan()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().tan())> == values::tan(2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.tan()}(), cdp2.tan()(0, 0)));
  EXPECT_EQ(0, cdp2.tan()(0, 1));
  static_assert(zero<decltype(z.tan())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().tan()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).tan())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().tan())>);
  static_assert(not hermitian_matrix<decltype(cxa.tan())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_acos_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_1>().acos())>, values::acos(1.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{(cp2/4).acos()}(), (cp2/4).acos()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.acos()}(), cxa.acos()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxb.acos()}(), cxb.acos()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_m1>().acos())>);
  static_assert(not zero<decltype(z.acos())>);
  static_assert(not triangular_matrix<decltype((std::declval<Tuv22>()* 0.25).array().acos()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id* 0.25).acos())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>()* 0.25).array().acos())>);
  static_assert(not hermitian_matrix<decltype((cxa* 0.25).acos())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_asin_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_1>().asin())>, values::asin(1.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{(cp2/4).asin()}(), (cp2/4).asin()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.asin()}(), cxa.asin()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxb.asin()}(), cxb.asin()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m1>().asin())> == values::asin(-1.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{(cdp2*0.25).asin()}(), (cdp2*0.25).asin()(0, 0)));
  EXPECT_EQ(0, (cdp2*0.25).asin()(0, 1));
  static_assert(zero<decltype(z.asin())>);
  static_assert(triangular_matrix<decltype((std::declval<Tuv22>() * 0.25).array().asin()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 0.25).asin())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>() * 0.25).array().asin())>);
  static_assert(not hermitian_matrix<decltype((cxa* 0.25).asin())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_atan_op)
{
  //static_assert(values::atan(11.) > 1.1);
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().atan())>, values::atan(2.)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cp2.atan()}(), cp2.atan()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.atan()}(), cxa.atan()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.atan()}(), cxb.atan()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_m2>().atan())> == values::atan(-2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{(cdp2*0.25).atan()}(), (cdp2*0.25).atan()(0, 0)));
  EXPECT_EQ(0, (cdp2*0.25).atan()(0, 1));
  static_assert(zero<decltype(z.atan())>);
  static_assert(triangular_matrix<decltype((std::declval<Tuv22>() * 0.25).array().atan()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 0.25).atan())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>() * 0.25).array().atan())>);
  static_assert(not hermitian_matrix<decltype((cxa * 0.25).atan())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_tanh_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().tanh())>, values::tanh(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.tanh()}(), cp2.tanh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.tanh()}(), cxa.tanh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.tanh()}(), cxb.tanh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().tanh())> == values::tanh(2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.tanh()}(), cdp2.tanh()(0, 0)));
  EXPECT_EQ(0, cdp2.tanh()(0, 1));
  static_assert(zero<decltype(z.tanh())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().tanh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).tanh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().tanh())>);
  static_assert(not hermitian_matrix<decltype(cxa.tanh())>); // because cxa is not necessarily hermitian
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_CwiseUnaryOp_scalar_atanh_op)
{
  EXPECT_TRUE(values::internal::near(constant_coefficient{(cp2 * 0.3).atanh()}(), (cp2 * 0.3).atanh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.atanh()}(), cxa.atanh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.atanh()}(), cxb.atanh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{(cdp2 * 0.3).atanh()}(), (cdp2 * 0.3).atanh()(0, 0)));
  EXPECT_EQ(0, (cdp2 * 0.3).atanh()(0, 1));
  static_assert(zero<decltype(z.atanh())>);
  static_assert(triangular_matrix<decltype((std::declval<Tuv22>() * 0.3).array().atanh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * -0.9).atanh())>);
  static_assert(hermitian_matrix<decltype((std::declval<Salv22>() * 0.3).array().atanh())>);
  static_assert(not hermitian_matrix<decltype(cxa.atanh())>); // because cxa is not necessarily hermitian
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_sinh_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().sinh())>, values::sinh(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.sinh()}(), cp2.sinh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.sinh()}(), cxa.sinh()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.sinh()}(), cxb.sinh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().sinh())> == values::sinh(2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.sinh()}(), cdp2.sinh()(0, 0)));
  EXPECT_EQ(0, cdp2.sinh()(0, 1));
  static_assert(zero<decltype(z.sinh())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().sinh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).sinh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().sinh())>);
  static_assert(not hermitian_matrix<decltype(cxa.sinh())>); // because cxa is not necessarily hermitian
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_CwiseUnaryOp_scalar_asinh_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().asinh())>, values::asinh(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.asinh()}(), cp2.asinh()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.asinh()}(), cxa.asinh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.asinh()}(), cxb.asinh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().asinh())> == values::asinh(2.));
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.asinh()}(), cdp2.asinh()(0, 0)));
  EXPECT_EQ(0, cdp2.asinh()(0, 1));
  static_assert(zero<decltype(z.asinh())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().asinh()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).asinh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().asinh())>);
  static_assert(not hermitian_matrix<decltype(cxa.asinh())>); // because cxa is not necessarily hermitian
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_cosh_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().cosh())>, values::cosh(2.)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.cosh()}(), cp2.cosh()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.cosh()}(), cxa.cosh()(0, 0)));
  EXPECT_TRUE(values::internal::near<100>(constant_coefficient{cxb.cosh()}(), cxb.cosh()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().cosh())>);
  static_assert(not zero<decltype(z.cosh())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().cosh()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).cosh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().cosh())>);
  static_assert(not hermitian_matrix<decltype(cxa.cosh())>); // because cxa is not necessarily hermitian
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_CwiseUnaryOp_scalar_acosh_op)
{
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cp2.acosh()}(), cp2.acosh()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxa.acosh()}(), cxa.acosh()(0, 0)));
  EXPECT_TRUE(values::internal::near<10>(constant_coefficient{cxb.acosh()}(), cxb.acosh()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().acosh())>);
  static_assert(not zero<decltype(z.acosh())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().acosh()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).acosh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().acosh())>);
  static_assert(not hermitian_matrix<decltype(cxa.acosh())>); // because cxa is not necessarily hermitian
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_inverse_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().inverse())>, 0.5));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.inverse()}(), cp2.inverse()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.inverse()}(), cxa.inverse()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().inverse())>);
  static_assert(not zero<decltype(z.inverse())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().inverse()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).inverse())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().inverse())>);
  static_assert(not hermitian_matrix<decltype(cxa.inverse())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_square_op)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().square())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().square())> == 4);
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.square()}(), cp2.square()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.square()}(), cxa.square()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.square()}(), cxb.square()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().square())> == 4);
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.square()}(), cdp2.square()(0, 0)));
  EXPECT_EQ(0, cdp2.square()(0, 1));
  static_assert(zero<decltype(z.square())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().square()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).square())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().square())>);
  static_assert(not hermitian_matrix<decltype(cxa.square())>); // because cxa is not necessarily hermitian
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_cube_op)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().cube())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().cube())> == -8);
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.cube()}(), cp2.cube()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.cube()}(), cxa.cube()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().cube())> == 8);
  EXPECT_TRUE(values::internal::near(constant_diagonal_coefficient{cdp2.cube()}(), cdp2.cube()(0, 0)));
  EXPECT_EQ(0, cdp2.cube()(0, 1));
  static_assert(zero<decltype(z.cube())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().cube()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((id * 3).cube())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().cube())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().cube())>);
  static_assert(not hermitian_matrix<decltype(cxa.cube())>); // because cxa is not necessarily hermitian
}

// Eigen::internal::scalar_round_op not implemented
// Eigen::internal::scalar_floor_op not implemented
// Eigen::internal::scalar_rint_op not implemented (Eigen 3.4+)
// Eigen::internal::scalar_ceil_op not implemented

// Eigen::internal::scalar_isnan_op not implemented
// Eigen::internal::scalar_isinf_op not implemented
// Eigen::internal::scalar_isfinite_op not implemented


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_boolean_not_op)
{
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_true>())> == false);
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_false>())> == true);
  static_assert(constant_coefficient_v<decltype(not std::declval<C22_1>())> == false); // requires narrowing from 1 to true.
  static_assert(constant_coefficient_v<decltype(not std::declval<Z22>())> == true); // requires narrowing from 0 to false.
  static_assert(zero<decltype(not std::declval<B22_true>())>);
  static_assert(zero<decltype(std::declval<B22_false>())>);

  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_true>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_false>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<C22_1>())> == false); // requires narrowing from 1 to true.
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<Z11>())> == true); // requires narrowing from 0 to false.

  static_assert(not constant_diagonal_matrix<decltype(not std::declval<B22_false>())>);
  static_assert(not diagonal_matrix<decltype(not std::declval<B22_false>())>);
  static_assert(hermitian_matrix<decltype(not std::declval<B22_false>())>);
}

  // Eigen::internal::scalar_sign_op not implemented


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_CwiseUnaryOp_scalar_logistic_op)
{
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_2>().logistic())>, 1 / (1 + (values::exp(-2)))));
  static_assert(values::internal::near(constant_coefficient_v<decltype(std::declval<C22_m2>().logistic())>, 1 / (1 + (values::exp(2)))));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cp2.logistic()}(), cp2.logistic()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxa.logistic()}(), cxa.logistic()(0, 0)));
  EXPECT_TRUE(values::internal::near(constant_coefficient{cxb.logistic()}(), cxb.logistic()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().logistic())>);
  static_assert(not zero<decltype(z.logistic())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tuv22>().array().logistic()), TriangleType::upper>);
  static_assert(not diagonal_matrix<decltype((id * 3).logistic())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().logistic())>);
  static_assert(not hermitian_matrix<decltype(cxa.logistic())>); // because cxa is not necessarily hermitian
}
#endif


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_bind1st_op)
{
  using CB1sum = Eigen::internal::bind1st_op<Eigen::internal::scalar_sum_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>{cp2, CB1sum{3}}}()), 5);
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cm2)>{cm2, CB1sum{6}}}()), 4);
  static_assert(values::dynamic<constant_coefficient<Eigen::CwiseUnaryOp<CB1sum, Z22>>>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB1sum, Z22>>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cdp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>, Applicability::permitted>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cm2)>>);

  using CB1prod = Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1prod, decltype(cm2)>{cm2, CB1prod{6}}}()), -12);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB1prod, decltype(cdp2)>{cdp2, CB1prod{3}}}()), 6);
  static_assert(values::dynamic<constant_diagonal_coefficient<Eigen::CwiseUnaryOp<CB1prod, I22>>>);
  static_assert(zero<Eigen::CwiseUnaryOp<CB1prod, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, I22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, DM2>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cp2)>, Applicability::permitted>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cm2)>>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_scalar_bind2nd_op)
{
  using CB2sum = Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>{cp2, CB2sum{3}}}()), 5);
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cm2)>{cm2, CB2sum{6}}}()), 4);
  static_assert(values::dynamic<constant_coefficient<Eigen::CwiseUnaryOp<CB2sum, Z22>>>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB2sum, Z22>>);
  static_assert(not constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cdp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>, Applicability::permitted>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cm2)>>);

  using CB2prod = Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>{cp2, CB2prod{3}}}()), 6);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB2prod, decltype(cdm2)>{cdm2, CB2prod{6}}}()), -12);
  static_assert(values::dynamic<constant_diagonal_coefficient<Eigen::CwiseUnaryOp<CB2prod, I22>>>);
  static_assert(zero<Eigen::CwiseUnaryOp<CB2prod, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, I22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, DM2>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>, Applicability::permitted>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cm2)>>);
}


TEST(eigen3, Eigen_CwiseUnaryOp_functor_composition)
{
  auto c11_2 = M11::Identity() + M11::Identity();

  auto negate_negate = Eigen3::functor_composition<std::negate<double>, std::negate<double>>{};
  static_assert(negate_negate(5.0) == 5.0);
  static_assert(constant_coefficient_v<const Eigen::CwiseUnaryOp<decltype(negate_negate), C11_2>> == 2);
  static_assert(constant_coefficient_v<const Eigen::CwiseUnaryOp<const decltype(negate_negate), C11_2>> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().unaryExpr(negate_negate))> == 2);
  EXPECT_EQ(constant_coefficient {c11_2.unaryExpr(negate_negate)}, 2);

  auto ident = [](const auto& xpr) { return xpr; };
  constexpr auto ident_ident = Eigen3::functor_composition {std::move(ident), std::move(ident)};
  static_assert(ident_ident(5.0) == 5.0);

  auto abs_negate = Eigen3::functor_composition<Eigen::internal::scalar_abs_op<double>, std::negate<double>> {};
  static_assert(abs_negate(-5.0) == 5.0);
  static_assert(abs_negate(5.0) == 5.0);
  static_assert(constant_coefficient_v<const Eigen::CwiseUnaryOp<decltype(abs_negate), C11_2>> == 2);
  EXPECT_EQ(constant_coefficient {c11_2.unaryExpr(abs_negate)}, 2);

#ifdef __cpp_concepts
  auto abs_negate2 = Eigen3::functor_composition {[](const auto& x){ return OpenKalman::values::abs(x); }, std::negate<double>{}};
  static_assert(abs_negate2(5.0) == 5.0);
  static_assert(constant_coefficient_v<const Eigen::CwiseUnaryOp<decltype(abs_negate2), C11_2>> == 2);
  EXPECT_EQ(constant_coefficient {c11_2.unaryExpr(abs_negate2)}, 2);
#endif

  auto opp_negate = Eigen3::functor_composition {Eigen::internal::scalar_opposite_op<double>{}, std::negate<double>{}};
  static_assert(opp_negate(5.0) == 5.0);
  static_assert(opp_negate(-5.0) == -5.0);
  static_assert(constant_coefficient_v<const Eigen::CwiseUnaryOp<decltype(opp_negate), C11_2>> == 2);
  EXPECT_EQ(constant_coefficient {c11_2.unaryExpr(opp_negate)}, 2);
}

