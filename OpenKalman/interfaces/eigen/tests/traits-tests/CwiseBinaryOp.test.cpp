/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_CwiseBinaryOp)
{
  auto id = I22 {2, 2}; // Identity
  auto cid = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<cdouble>, CA22> {2, 2};
  auto z = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2

  // general CwiseBinaryOp
  static_assert(self_contained<decltype(2 * std::declval<I22>() + std::declval<I22>())>);
  static_assert(not self_contained<decltype(2 * std::declval<I22>() + A22 {1, 2, 3, 4})>);
  static_assert(index_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I22>() + std::declval<I22>())>, 0> == 2);
  static_assert(index_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I22>() + std::declval<I22>())>, 1> == 2);
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

  static_assert(one_dimensional<decltype(std::declval<M11>() + std::declval<Mx1>())>);
  static_assert(one_dimensional<decltype(std::declval<M1x>() + std::declval<M11>())>);
  static_assert(one_dimensional<decltype(std::declval<M1x>() + std::declval<Mx1>())>);
  static_assert(one_dimensional<decltype(std::declval<I11>() + std::declval<Axx>())>);
  static_assert(one_dimensional<decltype(std::declval<DMx>() + std::declval<M1x>())>);
  static_assert(one_dimensional<decltype(std::declval<DMx>() + std::declval<Mx1>())>);
  static_assert(one_dimensional<decltype(std::declval<Axx>() + std::declval<A11>())>);
  static_assert(one_dimensional<decltype(std::declval<Mxx>() + std::declval<Mx1>()), Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<decltype(std::declval<Mxx>() + std::declval<Mx1>())>);

  static_assert(square_shaped<decltype(std::declval<M2x>() + std::declval<Mx2>())>);
  static_assert(not square_shaped<decltype(std::declval<Mxx>() + std::declval<Mx2>())>);
  static_assert(square_shaped<decltype(std::declval<Mxx>() + std::declval<Mx2>()), Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<decltype(std::declval<DMx>() + std::declval<DMx>())>);
  static_assert(square_shaped<decltype(std::declval<DMx>() + std::declval<Mxx>())>);

  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_3>())> == 5);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_m2>())> == 0);
  EXPECT_EQ((constant_coefficient{cxa + cxb}()), (std::complex<double>{4, 6}));
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() + M22::Constant(3).array()}(), 5);
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() + z}(), 2);
  EXPECT_EQ(constant_coefficient {z + M22::Constant(3).array()}(), 3);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() + std::declval<Cd22_3>())> == 5);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() + std::declval<C22_m2>())> == 0);
  EXPECT_EQ(constant_diagonal_coefficient {id + id}(), 2);
  EXPECT_EQ((constant_diagonal_coefficient {I11{1, 1} + M11::Constant(3).array()}()), 4);

  static_assert(zero<decltype(std::declval<C22_2>() + std::declval<C22_m2>())>);
  static_assert(zero<decltype(std::declval<C21_2>() + std::declval<C21_m2>())>);

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
  static_assert(constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx::ConstantReturnType, Mxx>, ConstantType::dynamic_constant>);

  static_assert(not constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx, Mxx::ConstantReturnType>>);
  static_assert(constant_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Mxx, Mxx::ConstantReturnType>, ConstantType::dynamic_constant>);

  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() * std::declval<C21_m2>())> == -4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() * std::declval<Z21>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>() * std::declval<C21_2>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<M21>().array() * std::declval<Z21>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>() * std::declval<M21>().array())> == 0);
  EXPECT_EQ((constant_coefficient{cxa * cxb}()), (std::complex<double>{-5, 10}));
  EXPECT_EQ((constant_coefficient{cp2 * cm2}()), -4);
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() * M22::Constant(3).array()}(), 6);
  EXPECT_EQ((constant_coefficient{z * M22::Constant(3).array()}()), 0);
  EXPECT_EQ((constant_coefficient{M22::Constant(2).array() * z}()), 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<Cd22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<A22>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<A22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>() * std::declval<Cd22_3>())> == 6); // no conjugate-product test
  static_assert(constant_diagonal_matrix<decltype(cdp2), ConstantType::dynamic_constant>);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2}()), 2);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * cdm2}()), -4);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * z}()), 0);
  EXPECT_EQ((constant_diagonal_coefficient{z * cdm2}()), 0);
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
  static_assert(diagonal_matrix<decltype(std::declval<Tuvxx>().array() * std::declval<Tlvxx>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tuvxx>().array() * std::declval<Tlvxx>().array())>);

  static_assert(triangular_matrix<decltype(std::declval<DW21>() * 3), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(2 * std::declval<DW21>()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array() * std::declval<Tlv22>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv2x>().array() * std::declval<Tlvx2>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlvxx>().array() * std::declval<Tlvxx>().array()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<DW21>() * 3), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(2 * std::declval<DW21>()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array() * std::declval<Tuv22>().array()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv2x>().array() * std::declval<Tuvx2>().array()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuvxx>().array() * std::declval<Tuvxx>().array()), TriangleType::upper>);

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
  static_assert(OpenKalman::interface::indexible_object_traits<Eigen::CwiseBinaryOp<CProd, Sauv22, DW21>>::is_hermitian == true);
  static_assert(OpenKalman::interface::indexible_object_traits<Eigen::CwiseBinaryOp<CProd, DW21, DW21>>::is_hermitian == true);

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
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_EQ>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, false));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LT>, decltype(cm2), decltype(cp2)>{cm2, cp2})>, true));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LE>, decltype(cp2), decltype(cp2)>{cp2, cp2})>, true));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_GT>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, true));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_GE>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, true));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_NEQ>, decltype(cm2), decltype(cm2)>{cm2, cm2})>, false));
  // No test for Eigen::internal::ComparisonName::cmp_UNORD
  static_assert(not diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_EQ>, M22, M22>>);
  static_assert(not triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LT>, M22, M22>>);

  // scalar_hypot_op
  using CWHYP = Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, decltype(cp2), decltype(cm2)>;
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(CWHYP{cp2, cm2})>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, C21_2, C21_m2>>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(internal::are_within_tolerance(constant_diagonal_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Cd22_2, Cd22_3>>, OpenKalman::internal::constexpr_sqrt(13.)));
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, DW21, DW21>>);
  static_assert(not diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, M21, DW21>>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Tlv22, Tlv22>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Tuv22, Tuv22>, TriangleType::upper>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Sauv22, Salv22>>);

  // scalar_pow_op
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(cp2.pow(cm2))>, 0.25));
  using M11_int = eigen_matrix_t<int, 1, 1>;
  using C11_3_int = decltype(M11_int::Identity() + M11_int::Identity() + M11_int::Identity());
  using C21_3_int = Eigen::Replicate<C11_3_int, 2, 1>;
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_3_int>().array().pow(std::declval<C21_3_int>().array()))>, 27));
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
  static_assert(zero<decltype(std::declval<C22_2>() - std::declval<C22_2>())>);
  static_assert(zero<decltype(std::declval<C21_2>() - std::declval<C21_2>())>);
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
  static_assert(hermitian_matrix<decltype(std::declval<Salvxx>() - std::declval<Sauvxx>()), Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<decltype(std::declval<DW21>() - std::declval<DW21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauvxx>() - std::declval<Sauvxx>()), Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>() - std::declval<Salv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>() - std::declval<Sauv22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salvxx>() - std::declval<Sauvxx>()), Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>() - std::declval<Salv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>() - std::declval<Sauv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sauv22>() - std::declval<Salv22>())> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Salv22>() - std::declval<Sauv22>())> == HermitianAdapterType::any);
  EXPECT_EQ((constant_coefficient{cxa - cxb}()), (std::complex<double>{-2, -2}));

  // scalar_quotient_op
  static_assert(constant_coefficient_v<decltype(std::declval<C11_3>() / std::declval<C11_m2>())> == -1.5);
  static_assert(constant_coefficient_v<decltype(std::declval<Z11>() / std::declval<C11_3>())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C11_3>() / std::declval<Z11>()), ConstantType::static_constant>); // divide by zero
  static_assert(constant_matrix<decltype(std::declval<C11_3>() / std::declval<Z11>()), ConstantType::dynamic_constant>); // divide by zero, but determined at runtime
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>() / std::declval<C21_m2>())> == -1);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>() / std::declval<C21_m2>())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C21_2>() / std::declval<Z21>())>); // divide by zero
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_3>() / std::declval<C11_m2>())> == -1.5);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z11>() / std::declval<C11_m2>())> == 0);
  static_assert(constant_diagonal_matrix<decltype(std::declval<C11_3>().array() / std::declval<Z11>().array()), ConstantType::dynamic_constant>); // divide by zero
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

