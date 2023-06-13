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

using numbers::pi;

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

  using CM11 = eigen_matrix_t<cdouble, 1, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM33 = eigen_matrix_t<cdouble, 3, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using A11 = Eigen::Array<double, 1, 1>; static_assert(one_by_one_matrix<A11>);
  using A10 = Eigen::Array<double, 1, Eigen::Dynamic>;
  using A01 = Eigen::Array<double, Eigen::Dynamic, 1>;
  using A00 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using A22 = Eigen::Array<double, 2, 2>;
  using A20 = Eigen::Array<double, 2, Eigen::Dynamic>;
  using A02 = Eigen::Array<double, Eigen::Dynamic, 2>;
  using A00 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using CA22 = Eigen::Array<cdouble, 2, 2>;

  using I11 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A11>; static_assert(constant_coefficient_v<I11> == 1);
  static_assert(one_by_one_matrix<I11, Likelihood::maybe>);
  using I10 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A10>; static_assert(constant_coefficient_v<I10> == 1);
  using I01 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A01>; static_assert(constant_coefficient_v<I01> == 1);
  using I00 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A00>; static_assert(constant_coefficient_v<I00> == 1);
  using I22 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A22>;
  using I20 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A20>;

  using Z11 = decltype(std::declval<I11>() - std::declval<I11>());
  using Z22 = decltype(std::declval<I22>() - std::declval<I22>());
  using Z21 = Eigen::Replicate<Z11, 2, 1>;
  using Z23 = Eigen::Replicate<Z11, 2, 3>;
  using Z12 = Eigen::Replicate<Z11, 1, 2>;
  using Z20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic>;
  using Z02 = Eigen::Replicate<Z11, Eigen::Dynamic, 2>;
  using Z00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic>;
  using Z10 = Eigen::Replicate<Z11, 1, Eigen::Dynamic>;
  using Z01 = Eigen::Replicate<Z11, Eigen::Dynamic, 1>;

  using C11_1 = I11;
  using C22_1 = Eigen::Replicate<C11_1, 2, 2>;
  using C21_1 = Eigen::Replicate<C11_1, 2, 1>;
  using C12_1 = Eigen::Replicate<C11_1, 1, 2>;
  using C20_1 = Eigen::Replicate<C11_1, 2, Eigen::Dynamic>;
  using C10_1 = Eigen::Replicate<C11_1, 1, Eigen::Dynamic>;
  using C01_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 1>;
  using C02_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 2>;
  using C00_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_2 = decltype(std::declval<I11>() + std::declval<I11>());
  using C22_2 = Eigen::Replicate<C11_2, 2, 2>;
  using C21_2 = Eigen::Replicate<C11_2, 2, 1>;
  using C12_2 = Eigen::Replicate<C11_2, 1, 2>;
  using C20_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using C02_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C10_2 = Eigen::Replicate<C11_2, 1, Eigen::Dynamic>;
  using C01_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 1>;
  using C00_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_3 = decltype(std::declval<I11>() + std::declval<I11>() + std::declval<I11>());
  using C21_3 = Eigen::Replicate<C11_3, 2, 1>;
  using C20_3 = Eigen::Replicate<C11_3, 2, Eigen::Dynamic>;
  using C01_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, 1>;
  using C00_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m1 = decltype(-std::declval<I11>());
  using C21_m1 = Eigen::Replicate<C11_m1, 2, 1>;
  using C20_m1 = Eigen::Replicate<C11_m1, 2, Eigen::Dynamic>;
  using C01_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, 1>;
  using C00_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m2 = decltype(-(std::declval<I11>() + std::declval<I11>()));
  using C22_m2 = Eigen::Replicate<C11_m2, 2, 2>;
  using C21_m2 = Eigen::Replicate<C11_m2, 2, 1>;
  using C20_m2 = Eigen::Replicate<C11_m2, 2, Eigen::Dynamic>;
  using C01_m2 = Eigen::Replicate<C11_m2, Eigen::Dynamic, 1>;
  using C00_m2 = Eigen::Replicate<C11_m2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_1_complex = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<cdouble>, Eigen::Array<cdouble, 1, 1>>;
  using C11_2_complex = decltype(std::declval<C11_1_complex>() + std::declval<C11_1_complex>());

  using B11 = Eigen::Array<bool, 1, 1>;
  using B22 = Eigen::Array<bool, 2, 2>;

  using B11_true = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<bool>, B11>;
  using B11_false = decltype(not std::declval<B11_true>());
  using B22_true = decltype(std::declval<B11_true>().replicate<2,2>());
  using B22_false = decltype(std::declval<B11_false>().replicate<2,2>());
  using BI22 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<bool>, B22>;

  using Cd22_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using Cd20_2 = Eigen::Replicate<Cd22_2, 1, Eigen::Dynamic>;
  using Cd02_2 = Eigen::Replicate<Cd22_2, Eigen::Dynamic, 1>;
  using Cd00_2 = Eigen::Replicate<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_3 = decltype(std::declval<I22>() + std::declval<I22>() + std::declval<I22>());
  using Cd20_3 = Eigen::Replicate<Cd22_3, 1, Eigen::Dynamic>;
  using Cd02_3 = Eigen::Replicate<Cd22_3, Eigen::Dynamic, 1>;
  using Cd00_3 = Eigen::Replicate<Cd22_3, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_m1 = decltype(-std::declval<I22>());
  using Cd20_m1 = Eigen::Replicate<Cd22_m1, 1, Eigen::Dynamic>;
  using Cd02_m1 = Eigen::Replicate<Cd22_m1, Eigen::Dynamic, 1>;
  using Cd00_m1 = Eigen::Replicate<Cd22_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_m2 = decltype(-std::declval<Cd22_2>());
  using Cd20_m2 = Eigen::Replicate<Cd22_m2, 1, Eigen::Dynamic>;
  using Cd02_m2 = Eigen::Replicate<Cd22_m2, Eigen::Dynamic, 1>;
  using Cd00_m2 = Eigen::Replicate<Cd22_m2, Eigen::Dynamic, Eigen::Dynamic>;

  using DW2_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using DW0_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using DW2_m2 = decltype(-std::declval<DW2_2>());

  using DM2 = EigenWrapper<Eigen::DiagonalMatrix<double, 2>>;
  using DM0 = EigenWrapper<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>;

  using Md21 = EigenWrapper<Eigen::DiagonalWrapper<M21>>;
  using Md20_1 = EigenWrapper<Eigen::DiagonalWrapper<M20>>;
  using Md01_2 = EigenWrapper<Eigen::DiagonalWrapper<M01>>;
  using Md00_21 = EigenWrapper<Eigen::DiagonalWrapper<M00>>;

  using Salv22 = EigenWrapper<Eigen::SelfAdjointView<M22, Eigen::Lower>>;
  using Salv20 = EigenWrapper<Eigen::SelfAdjointView<M20, Eigen::Lower>>;
  using Salv02 = EigenWrapper<Eigen::SelfAdjointView<M02, Eigen::Lower>>;
  using Salv00 = EigenWrapper<Eigen::SelfAdjointView<M00, Eigen::Lower>>;

  using Sauv22 = EigenWrapper<Eigen::SelfAdjointView<M22, Eigen::Upper>>;
  using Sauv20 = EigenWrapper<Eigen::SelfAdjointView<M20, Eigen::Upper>>;
  using Sauv02 = EigenWrapper<Eigen::SelfAdjointView<M02, Eigen::Upper>>;
  using Sauv00 = EigenWrapper<Eigen::SelfAdjointView<M00, Eigen::Upper>>;

  using Sadv22 = EigenWrapper<Eigen::SelfAdjointView<M22::IdentityReturnType, Eigen::Lower>>;
  using Sadv20 = EigenWrapper<Eigen::SelfAdjointView<M20::IdentityReturnType, Eigen::Lower>>;
  using Sadv02 = EigenWrapper<Eigen::SelfAdjointView<M02::IdentityReturnType, Eigen::Lower>>;
  using Sadv00 = EigenWrapper<Eigen::SelfAdjointView<M00::IdentityReturnType, Eigen::Lower>>;

  using Tlv22 = EigenWrapper<Eigen::TriangularView<M22, Eigen::Lower>>;
  using Tlv20 = EigenWrapper<Eigen::TriangularView<M20, Eigen::Lower>>;
  using Tlv02 = EigenWrapper<Eigen::TriangularView<M02, Eigen::Lower>>;
  using Tlv00 = EigenWrapper<Eigen::TriangularView<M00, Eigen::Lower>>;

  using Tuv22 = EigenWrapper<Eigen::TriangularView<M22, Eigen::Upper>>;
  using Tuv20 = EigenWrapper<Eigen::TriangularView<M20, Eigen::Upper>>;
  using Tuv02 = EigenWrapper<Eigen::TriangularView<M02, Eigen::Upper>>;
  using Tuv00 = EigenWrapper<Eigen::TriangularView<M00, Eigen::Upper>>;
}


TEST(eigen3, Eigen_PartialReduxExpr)
{
  auto c22_m2 = (-(M11::Identity() + M11::Identity())).replicate<2,2>();
  auto c00_22_m2 = (-(M11::Identity() + M11::Identity())).replicate(2,2);
  auto c00_21_2 = (M11::Identity() + M11::Identity()).replicate(2,1);

  auto cd22_m2 = (-(M22::Identity()+M22::Identity()));
  auto cd20_2_m2 = Eigen::Replicate<decltype(cd22_m2), 1, Eigen::Dynamic> {cd22_m2, 1, 1};
  auto cd02_2_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, 1> {cd22_m2, 1, 1};
  auto cd00_22_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, Eigen::Dynamic> {cd22_m2, 1, 1};

  auto cd22_2 = (M22::Identity() + M22::Identity()).array();
  auto cd20_2_2 = Eigen::Replicate<decltype(cd22_2), 1, Eigen::Dynamic> {cd22_2, 1, 1};
  auto cd02_2_2 = Eigen::Replicate<decltype(cd22_2), Eigen::Dynamic, 1> {cd22_2, 1, 1};
  auto cd00_22_2 = Eigen::Replicate<decltype(cd22_2), Eigen::Dynamic, Eigen::Dynamic> {cd22_2, 1, 1};

  auto cd3322_m2 = cd22_m2.replicate<3,3>();
  auto cd3300_22_m2 = cd00_22_m2.replicate<3,3>();
  auto cd0022_33_m2 = cd22_m2.replicate(3, 3);

  auto cxb = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);

  // LpNorm<1>

  static_assert(interface::detail::SingleConstantPartialRedux<decltype(c22_m2), EGI::member_lpnorm<1, double, double>>::
    get_constant(constant_coefficient{c22_m2}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}) == 4);

  static_assert(constant_matrix<decltype(c22_m2), CompileTimeStatus::known>);
  static_assert(not zero_matrix<decltype(c22_m2)>);
  static_assert(not one_by_one_matrix<decltype(c22_m2), Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<decltype(c22_m2), CompileTimeStatus::any, Likelihood::maybe>);

  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<1>())> == 4);
  EXPECT_EQ(constant_coefficient{c22_m2.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(c00_22_m2.colwise().lpNorm<1>()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(c22_m2.rowwise().lpNorm<1>())> == 4);
  EXPECT_EQ(constant_coefficient{c22_m2.rowwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{c00_22_m2.rowwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(c00_22_m2.rowwise().lpNorm<1>()(0,0), 4);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<1>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().lpNorm<1>())> == 2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().lpNorm<1>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(cd20_2_m2.rowwise().lpNorm<1>()(0,0), 2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().lpNorm<1>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(cd02_2_m2.colwise().lpNorm<1>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().lpNorm<1>())> == 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().lpNorm<1>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(cd00_22_m2.colwise().lpNorm<1>()(0,0), 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().lpNorm<1>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(cd00_22_m2.rowwise().lpNorm<1>()(0,0), 2);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.rowwise().lpNorm<1>())> == 6);
  EXPECT_EQ(constant_coefficient{cd3300_22_m2.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cd3300_22_m2.rowwise().lpNorm<1>()}(), 6);
  static_assert(constant_matrix<decltype(cd0022_33_m2.colwise().lpNorm<1>()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(cd0022_33_m2.rowwise().lpNorm<1>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.rowwise().lpNorm<1>()}(), 6);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<1>())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().lpNorm<1>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<1>()(0,0), 0);

  // lpNorm<2>

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<2>())>, constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{c22_m2.colwise().lpNorm<2>()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(constant_coefficient{c00_22_m2.colwise().lpNorm<2>()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(c00_22_m2.colwise().lpNorm<2>()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<2>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<2>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().lpNorm<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().lpNorm<2>())> == 2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().lpNorm<2>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd20_2_m2.rowwise().lpNorm<2>()(0,0), 2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().lpNorm<2>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd02_2_m2.colwise().lpNorm<2>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().lpNorm<2>())> == 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().lpNorm<2>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd00_22_m2.colwise().lpNorm<2>()(0,0), 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().lpNorm<2>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd00_22_m2.rowwise().lpNorm<2>()(0,0), 2);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().lpNorm<2>())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().lpNorm<2>())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.colwise().lpNorm<2>())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.rowwise().lpNorm<2>())> == 2 * numbers::sqrt3);
  EXPECT_EQ(constant_coefficient{cd3300_22_m2.colwise().lpNorm<2>()}(), 2 * numbers::sqrt3);
  EXPECT_EQ(constant_coefficient{cd3300_22_m2.rowwise().lpNorm<2>()}(), 2 * numbers::sqrt3);
  static_assert(constant_matrix<decltype(cd0022_33_m2.colwise().lpNorm<2>()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(cd0022_33_m2.rowwise().lpNorm<2>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.colwise().lpNorm<2>()}(), 2 * numbers::sqrt3);
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.rowwise().lpNorm<2>()}(), 2 * numbers::sqrt3);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().lpNorm<2>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<2>()(0,0), 0);

  // lpNorm<3>

  static_assert(are_within_tolerance<5>(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<3>())>, constexpr_pow(16., 1./3)));
  EXPECT_NEAR(constant_coefficient{c22_m2.colwise().lpNorm<3>()}(), (std::pow(16., 1./3)), 1e-9);
  EXPECT_NEAR(constant_coefficient{c00_22_m2.colwise().lpNorm<3>()}(), (std::pow(16., 1./3)), 1e-9);
  EXPECT_NEAR(c00_22_m2.colwise().lpNorm<3>()(0,0), (std::pow(16., 1./3)), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<3>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<3>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<3>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().lpNorm<3>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().lpNorm<3>())> == 2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().lpNorm<3>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd20_2_m2.rowwise().lpNorm<3>()(0,0), 2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().lpNorm<3>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd02_2_m2.colwise().lpNorm<3>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().lpNorm<3>())> == 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().lpNorm<3>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd00_22_m2.colwise().lpNorm<3>()(0,0), 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().lpNorm<3>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd00_22_m2.rowwise().lpNorm<3>()(0,0), 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<3>())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().lpNorm<3>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<3>()(0,0), 0);

  // lpNorm<Eigen::Infinity>

  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<Eigen::Infinity>())> == 2);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().lpNorm<Eigen::Infinity>()}(), 2);
  EXPECT_EQ(c00_22_m2.colwise().lpNorm<Eigen::Infinity>()(0,0), 2);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<Eigen::Infinity>())> == 2);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().lpNorm<Eigen::Infinity>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<Eigen::Infinity>()(0,0), 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<Eigen::Infinity>())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().lpNorm<Eigen::Infinity>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<Eigen::Infinity>()(0,0), 0);

  // lpNorm<0>

  static_assert(constant_matrix<decltype(c00_22_m2.colwise().lpNorm<0>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(c00_22_m2.colwise().lpNorm<0>()(0,0), INFINITY);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<0>()(0,0), INFINITY);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().lpNorm<0>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd20_2_m2.rowwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().lpNorm<0>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd02_2_m2.colwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().lpNorm<0>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd00_22_m2.colwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().lpNorm<0>()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd00_22_m2.rowwise().lpNorm<0>()(0,0), INFINITY);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<0>())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().lpNorm<0>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<0>()(0,0), INFINITY);

  // stableNorm

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().stableNorm())>, constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{c00_22_m2.colwise().stableNorm()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(c00_22_m2.colwise().stableNorm()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().stableNorm()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().stableNorm()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().stableNorm()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().stableNorm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().stableNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().stableNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().stableNorm())> == 2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().stableNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().stableNorm()}(), 2);
  EXPECT_EQ(cd20_2_m2.rowwise().stableNorm()(0,0), 2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().stableNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().stableNorm()}(), 2);
  EXPECT_EQ(cd02_2_m2.colwise().stableNorm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().stableNorm())> == 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().stableNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().stableNorm()}(), 2);
  EXPECT_EQ(cd00_22_m2.colwise().stableNorm()(0,0), 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().stableNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().stableNorm()}(), 2);
  EXPECT_EQ(cd00_22_m2.rowwise().stableNorm()(0,0), 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().stableNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().stableNorm())>);

  // hypotNorm

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().hypotNorm())>, constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{c00_22_m2.colwise().hypotNorm()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(c00_22_m2.colwise().hypotNorm()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().hypotNorm()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().hypotNorm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().hypotNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().hypotNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().hypotNorm())> == 2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().hypotNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd20_2_m2.rowwise().hypotNorm()(0,0), 2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().hypotNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd02_2_m2.colwise().hypotNorm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().hypotNorm())> == 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().hypotNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd00_22_m2.colwise().hypotNorm()(0,0), 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().hypotNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd00_22_m2.rowwise().hypotNorm()(0,0), 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().hypotNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().hypotNorm())>);

  // sum

  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().sum()}(), -4);
  EXPECT_EQ(c00_22_m2.colwise().sum()(0,0), -4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().sum())> == 4);
  static_assert(constant_matrix<decltype(std::declval<C20_2>().rowwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<C02_2>().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().sum())> == 4);
  static_assert(constant_matrix<decltype(std::declval<C00_2>().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<C00_2>().rowwise().sum()), CompileTimeStatus::unknown>);

  static_assert(constant_matrix<decltype(std::declval<Cd00_m2>().colwise().sum()), CompileTimeStatus::unknown>);

  EXPECT_EQ(constant_coefficient{c00_21_2.replicate(2,2).colwise().sum()}(), 8);
  EXPECT_EQ(c00_21_2.replicate(2,2).colwise().sum()(0,0), 8);
  EXPECT_EQ(constant_coefficient{c00_21_2.replicate(2,2).rowwise().sum()}(), 4);
  EXPECT_EQ(c00_21_2.replicate(2,2).rowwise().sum()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().rowwise().sum())> == 2);
  static_assert(constant_matrix<decltype(std::declval<C01_2>().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<C01_2>().rowwise().sum())> == 2);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().sum()}(), -2);
  EXPECT_EQ(cd22_m2.colwise().sum()(0,0), -2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().sum()}(), -2);
  EXPECT_EQ(cd22_m2.rowwise().sum()(0,0), -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().sum())> == -2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().sum()}(), -2);
  EXPECT_EQ(cd20_2_m2.rowwise().sum()(0,0), -2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().sum()}(), -2);
  EXPECT_EQ(cd02_2_m2.colwise().sum()(0,0), -2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().sum())> == -2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().sum()}(), -2);
  EXPECT_EQ(cd00_22_m2.colwise().sum()(0,0), -2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().sum()}(), -2);
  EXPECT_EQ(cd00_22_m2.rowwise().sum()(0,0), -2);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().sum())> == -6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().sum())> == -6);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.colwise().sum())> == -6);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.rowwise().sum())> == -6);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().sum())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().sum())>);

  EXPECT_EQ(constant_coefficient{cxb.colwise().sum().imag()}(), 8);
  EXPECT_EQ(constant_coefficient{cxb.rowwise().sum().imag()}(), 8);
  EXPECT_EQ(cxb.colwise().sum().imag()(0,0), 8);
  EXPECT_EQ(cxb.rowwise().sum().imag()(0,0), 8);
  EXPECT_EQ(constant_coefficient{Eigen3::EigenWrapper{cxb.array().matrix()}.colwise().sum().imag()}(), 8);
  EXPECT_EQ(constant_coefficient{Eigen3::EigenWrapper{cxb.array().matrix()}.rowwise().sum().imag()}(), 8);
  EXPECT_EQ(cxb.array().matrix().colwise().sum().imag()(0,0), 8);
  EXPECT_EQ(cxb.array().matrix().rowwise().sum().imag()(0,0), 8);
  EXPECT_EQ(Eigen3::EigenWrapper{cxb.array().matrix()}.colwise().sum().imag()(0,0), 8);
  EXPECT_EQ(Eigen3::EigenWrapper{cxb.array().matrix()}.rowwise().sum().imag()(0,0), 8);

  EXPECT_EQ(constant_coefficient{Eigen3::EigenWrapper{cxb.array().matrix()}.imag().colwise().sum()}(), 8);
  EXPECT_EQ(constant_coefficient{Eigen3::EigenWrapper{cxb.array().matrix()}.imag().rowwise().sum()}(), 8);
  EXPECT_EQ(cxb.array().matrix().imag().colwise().sum()(0,0), 8);
  EXPECT_EQ(cxb.array().matrix().imag().rowwise().sum()(0,0), 8);
  EXPECT_EQ(Eigen3::EigenWrapper{cxb.array().matrix()}.imag().colwise().sum()(0,0), 8);
  EXPECT_EQ(Eigen3::EigenWrapper{cxb.array().matrix()}.imag().rowwise().sum()(0,0), 8);

  static_assert(constant_coefficient_v<decltype(c22_m2.cwiseAbs2().colwise().sum())> == 8);
  static_assert(constant_matrix<decltype(c00_22_m2.cwiseAbs2().rowwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{c00_22_m2.cwiseAbs2().colwise().sum()}(), 8);

  static_assert(constant_coefficient_v<decltype(cd22_m2.cwiseAbs2().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(Eigen3::EigenWrapper{cd22_m2.array().matrix()}.cwiseAbs2().colwise().sum())> == 4);
  static_assert(constant_matrix<decltype(std::declval<Cd00_m2>().abs2().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<Cd00_m2>().abs2().rowwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(cd00_22_m2.cwiseAbs2().colwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.cwiseAbs2().colwise().sum()}(), 4);
  EXPECT_EQ(cd00_22_m2.cwiseAbs2().colwise().sum()(0,0), 4);
  EXPECT_EQ(constant_coefficient{Eigen3::EigenWrapper{cxb.array().matrix()}.cwiseAbs2().colwise().sum()}(), 50);
  EXPECT_EQ(Eigen3::EigenWrapper{cxb.array().matrix()}.cwiseAbs2().colwise().sum()(0,0), 50);

  // squaredNorm -- Note: Eigen version 3.4 calculates x._wise().squaredNorm() as if it were x.cwiseAbs2()._wise().sum().

  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().squaredNorm())> == 8);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().squaredNorm()}(), 8);
  EXPECT_EQ(c00_22_m2.colwise().squaredNorm()(0,0), 8);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd22_m2.colwise().squaredNorm()(0,0), 4);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd22_m2.rowwise().squaredNorm()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().squaredNorm())> == 4);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().squaredNorm())> == 4);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().squaredNorm())> == 4);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().squaredNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd20_2_m2.rowwise().squaredNorm()(0,0), 4);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().squaredNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd02_2_m2.colwise().squaredNorm()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().squaredNorm())> == 4);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().squaredNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd00_22_m2.colwise().squaredNorm()(0,0), 4);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().squaredNorm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd00_22_m2.rowwise().squaredNorm()(0,0), 4);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().squaredNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().squaredNorm())>);

  // norm -- Note: Eigen version 3.4 calculates x._wise().norm() as if it were x.cwiseAbs2()._wise().sum().sqrt().

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().norm())>, constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{c00_22_m2.colwise().norm()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(c00_22_m2.colwise().norm()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().norm()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().norm()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().norm()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().norm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().norm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().norm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().norm())> == 2);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().norm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().norm()}(), 2);
  EXPECT_EQ(cd20_2_m2.rowwise().norm()(0,0), 2);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().norm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().norm()}(), 2);
  EXPECT_EQ(cd02_2_m2.colwise().norm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().norm())> == 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.colwise().norm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().norm()}(), 2);
  EXPECT_EQ(cd00_22_m2.colwise().norm()(0,0), 2);
  static_assert(constant_matrix<decltype(cd00_22_m2.rowwise().norm()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().norm()}(), 2);
  EXPECT_EQ(cd00_22_m2.rowwise().norm()(0,0), 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().norm())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().norm())>);

  // redux \todo

  //EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})}(), -4);
  EXPECT_EQ(c00_22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})(0,0), -4);
  //EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().redux(std::plus<double>{})}(), -4);
  EXPECT_EQ(c00_22_m2.colwise().redux(std::plus<double>{})(0,0), -4);
  //EXPECT_EQ(constant_coefficient{c00_22_m2.rowwise().redux(Eigen::internal::scalar_product_op<double, double>{})}(), 4);
  EXPECT_EQ(c00_22_m2.rowwise().redux(Eigen::internal::scalar_product_op<double, double>{})(0,0), 4);
  //EXPECT_EQ(constant_coefficient{c00_22_m2.rowwise().redux(std::multiplies<double>{})}(), 4);
  EXPECT_EQ(c00_22_m2.rowwise().redux(std::multiplies<double>{})(0,0), 4);

  // mean -- Note: Eigen version 3.4 calculates x._wise.mean() as if it were x._wise.sum() / dimension.

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  static_assert(constant_matrix<decltype(cd22_2.colwise().mean()), CompileTimeStatus::unknown>);

  EXPECT_EQ(get_scalar_constant_value(constant_coefficient{(M22::Identity() - M22::Identity()).colwise().mean()}), 0.);
  EXPECT_EQ(get_scalar_constant_value(constant_coefficient{M22::Zero().colwise().mean()}), 0.);
#else
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().mean())> == 2);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().mean())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.rowwise().mean())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().mean())> == -2);
  static_assert(not constant_matrix<decltype(cd02_2_m2.colwise().mean())>);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().mean())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().mean())>);
#endif

  // minCoeff

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C22_2>().rowwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().rowwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C02_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C00_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C00_2>().rowwise().minCoeff())> == 2);

  EXPECT_EQ(constant_coefficient{cd22_2.colwise().minCoeff()}(), 0);
  EXPECT_EQ(cd22_2.colwise().minCoeff()(0,0), 0);
  static_assert(constant_coefficient_v<decltype(cd22_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd22_2.rowwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd20_2_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd20_2_2.rowwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd02_2_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd02_2_2.rowwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd00_22_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd00_22_2.rowwise().minCoeff())> == 0);
  EXPECT_EQ(constant_coefficient{cd20_2_2.rowwise().minCoeff()}(), 0);
  EXPECT_EQ(cd20_2_2.rowwise().minCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cd02_2_2.colwise().minCoeff()}(), 0);
  EXPECT_EQ(cd02_2_2.colwise().minCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cd00_22_2.rowwise().minCoeff()}(), 0);
  EXPECT_EQ(cd00_22_2.rowwise().minCoeff()(0,0), 0);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().minCoeff()}(), -2);
  EXPECT_EQ(cd22_m2.colwise().minCoeff()(0,0), -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.rowwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd00_22_m2.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd00_22_m2.rowwise().minCoeff())> == -2);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().minCoeff()}(), -2);
  EXPECT_EQ(cd20_2_m2.rowwise().minCoeff()(0,0), -2);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().minCoeff()}(), -2);
  EXPECT_EQ(cd02_2_m2.colwise().minCoeff()(0,0), -2);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.rowwise().minCoeff()}(), -2);
  EXPECT_EQ(cd00_22_m2.rowwise().minCoeff()(0,0), -2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().minCoeff())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().minCoeff())>);

  // maxCoeff

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C02_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C00_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C00_2>().rowwise().maxCoeff())> == 2);

  EXPECT_EQ(constant_coefficient{cd22_2.colwise().maxCoeff()}(), 2);
  EXPECT_EQ(cd22_2.colwise().maxCoeff()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_2.colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_2.colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd20_2_2.rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_2.colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd02_2_2.rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd00_22_2.colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd00_22_2.rowwise().maxCoeff())> == 2);
  EXPECT_EQ(constant_coefficient{cd20_2_2.rowwise().maxCoeff()}(), 2);
  EXPECT_EQ(cd20_2_2.rowwise().maxCoeff()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd02_2_2.colwise().maxCoeff()}(), 2);
  EXPECT_EQ(cd02_2_2.colwise().maxCoeff()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd00_22_2.colwise().maxCoeff()}(), 2);
  EXPECT_EQ(cd00_22_2.colwise().maxCoeff()(0,0), 2);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd22_m2.colwise().maxCoeff()(0,0), 0);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd00_22_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd00_22_m2.rowwise().maxCoeff())> == 0);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd20_2_m2.rowwise().maxCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd02_2_m2.colwise().maxCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd00_22_m2.colwise().maxCoeff()(0,0), 0);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().maxCoeff())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().maxCoeff())>);

  // all

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_false>().colwise().all())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>().colwise().all())> == false);

  // any

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_true>().colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_false>().colwise().any())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>().colwise().any())> == true);

  // count

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C12_2>().colwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<const C20_2>().colwise().count())> == 2);
  static_assert(constant_matrix<decltype(std::declval<C02_2>().colwise().count()), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(std::declval<C02_2>().colwise().count()), CompileTimeStatus::known>);

  static_assert(constant_coefficient_v<decltype(std::declval<const C22_2>().rowwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C21_2>().rowwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().count())> == 2);
  static_assert(constant_matrix<decltype(std::declval<C20_2>().rowwise().count()), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(std::declval<C20_2>().rowwise().count()), CompileTimeStatus::known>);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().count())> == 1);
  static_assert(constant_matrix<decltype(cd02_2_m2.colwise().count()), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(cd02_2_m2.colwise().count()), CompileTimeStatus::known, Likelihood::maybe>);
  EXPECT_EQ(constant_coefficient{cd02_2_m2.colwise().count()}(), 1);
  EXPECT_EQ(cd02_2_m2.colwise().count()(0,0), 1);

  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(cd02_2_m2.rowwise().count())> == 1);
  static_assert(constant_matrix<decltype(cd20_2_m2.rowwise().count()), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(cd20_2_m2.rowwise().count()), CompileTimeStatus::known, Likelihood::maybe>);
  EXPECT_EQ(constant_coefficient{cd20_2_m2.rowwise().count()}(), 1);
  EXPECT_EQ(cd20_2_m2.rowwise().count()(0,0), 1);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().count())> == 3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().count())> == 3);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.colwise().count())> == 3);
  static_assert(constant_coefficient_v<decltype(cd3300_22_m2.rowwise().count())> == 3);

  // prod

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().prod())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().prod())> == 4);
  static_assert(constant_matrix<decltype(std::declval<C20_2>().rowwise().prod()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<C02_2>().colwise().prod()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().prod())> == 4);

  static_assert(zero_matrix<decltype(cd22_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd20_2_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd20_2_m2.rowwise().prod())>);
  static_assert(zero_matrix<decltype(cd02_2_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd02_2_m2.rowwise().prod())>);
  EXPECT_EQ(cd20_2_m2.rowwise().prod()(0,0), 0);
  EXPECT_EQ(cd02_2_m2.colwise().prod()(0,0), 0);

  static_assert(zero_matrix<decltype(cd3322_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd3322_m2.rowwise().prod())>);
  static_assert(zero_matrix<decltype(cd3300_22_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd3300_22_m2.rowwise().prod())>);
  EXPECT_EQ(cd3300_22_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cd3300_22_m2.rowwise().prod()(0,0), 0);
  static_assert(zero_matrix<decltype(cd0022_33_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd0022_33_m2.rowwise().prod())>);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().prod())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().prod())>);

  // reverse

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().reverse())> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().colwise().reverse())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().rowwise().reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd22_2.reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd22_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd20_2_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd02_2_m2.reverse())> == -2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().reverse())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().reverse())>);

  // replicate

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate(2))> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(cd22_2.colwise().replicate<1>())> == 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().replicate<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().replicate<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().replicate(2))>);
  static_assert(zero_matrix<decltype(std::declval<Z00>().colwise().replicate(2))>);
}


TEST(eigen3, Eigen_VectorWiseOp)
{
  static_assert(max_indices_of_v<decltype(std::declval<M34>().colwise())> == 2);
  static_assert(std::is_same_v<scalar_type_of_t<decltype(std::declval<M34>().colwise())>, double>);

  static_assert(index_dimension_of_v<decltype(std::declval<M34>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M30>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M34>().colwise()), 1> == 4);
  static_assert(index_dimension_of_v<decltype(std::declval<M04>().colwise()), 1> == 4);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise())> == 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().rowwise())>);
}
