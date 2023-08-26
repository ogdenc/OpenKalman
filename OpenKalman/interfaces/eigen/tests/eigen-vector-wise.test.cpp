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
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;


TEST(eigen3, Eigen_PartialReduxExpr_norms)
{
  auto c22_m2 = (-(M11::Identity() + M11::Identity())).replicate<2,2>();
  auto c00_22_m2 = (-(M11::Identity() + M11::Identity())).replicate(2,2);

  auto cd22_m2 = (-(M22::Identity()+M22::Identity()));
  auto cd20_2_m2 = Eigen::Replicate<decltype(cd22_m2), 1, Eigen::Dynamic> {cd22_m2, 1, 1};
  auto cd02_2_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, 1> {cd22_m2, 1, 1};
  auto cd00_22_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, Eigen::Dynamic> {cd22_m2, 1, 1};

  auto cd3322_m2 = cd22_m2.replicate<3,3>();
  auto cd3300_22_m2 = cd00_22_m2.replicate<3,3>();
  auto cd0022_33_m2 = cd22_m2.replicate(3, 3);

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);

  // LpNorm<1>

  static_assert(Eigen3::SingleConstantPartialRedux<decltype(c22_m2), Eigen::internal::member_lpnorm<1, double, double>>::
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().lpNorm<1>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<1>()(0,0), 0);

  // lpNorm<2>

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<2>())>, internal::constexpr_sqrt(8.)));
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().lpNorm<2>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<2>()(0,0), 0);

  // lpNorm<3>

  static_assert(are_within_tolerance<5>(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<3>())>, internal::constexpr_pow(16., 1./3)));
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().lpNorm<3>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<3>()(0,0), 0);

  // lpNorm<Eigen::Infinity>

  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<Eigen::Infinity>())> == 2);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().lpNorm<Eigen::Infinity>()}(), 2);
  EXPECT_EQ(c00_22_m2.colwise().lpNorm<Eigen::Infinity>()(0,0), 2);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<Eigen::Infinity>())> == 2);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.colwise().lpNorm<Eigen::Infinity>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<Eigen::Infinity>()(0,0), 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<Eigen::Infinity>())>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().lpNorm<Eigen::Infinity>())>);
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().lpNorm<0>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<0>()(0,0), INFINITY);

  // stableNorm

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().stableNorm())>, internal::constexpr_sqrt(8.)));
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().stableNorm())>);

  // hypotNorm

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().hypotNorm())>, internal::constexpr_sqrt(8.)));
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().hypotNorm())>);

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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().squaredNorm())>);

  // norm -- Note: Eigen version 3.4 calculates x._wise().norm() as if it were x.cwiseAbs2()._wise().sum().sqrt().

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().norm())>, internal::constexpr_sqrt(8.)));
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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().norm())>);
}


TEST(eigen3, Eigen_PartialReduxExpr_sum)
{
  auto c22_m2 = (-(M11::Identity() + M11::Identity())).replicate<2,2>();
  auto c00_22_m2 = (-(M11::Identity() + M11::Identity())).replicate(2,2);
  auto c00_21_2 = (M11::Identity() + M11::Identity()).replicate(2,1);

  auto cd22_m2 = (-(M22::Identity()+M22::Identity()));
  auto cd20_2_m2 = Eigen::Replicate<decltype(cd22_m2), 1, Eigen::Dynamic> {cd22_m2, 1, 1};
  auto cd02_2_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, 1> {cd22_m2, 1, 1};
  auto cd00_22_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, Eigen::Dynamic> {cd22_m2, 1, 1};

  auto cd3322_m2 = cd22_m2.replicate<3,3>();
  auto cd3300_22_m2 = cd00_22_m2.replicate<3,3>();
  auto cd0022_33_m2 = cd22_m2.replicate(3, 3);

  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);

  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().sum()}(), -4);
  EXPECT_EQ(c00_22_m2.colwise().sum()(0,0), -4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().sum())> == 4);
  static_assert(constant_matrix<decltype(std::declval<C2x_2>().rowwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<Cx2_2>().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().sum())> == 4);
  static_assert(constant_matrix<decltype(std::declval<Cxx_2>().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<Cxx_2>().rowwise().sum()), CompileTimeStatus::unknown>);

  static_assert(constant_matrix<decltype(std::declval<Cdxx_m2>().colwise().sum()), CompileTimeStatus::unknown>);

  EXPECT_EQ(constant_coefficient{c00_21_2.replicate(2,2).colwise().sum()}(), 8);
  EXPECT_EQ(c00_21_2.replicate(2,2).colwise().sum()(0,0), 8);
  EXPECT_EQ(constant_coefficient{c00_21_2.replicate(2,2).rowwise().sum()}(), 4);
  EXPECT_EQ(c00_21_2.replicate(2,2).rowwise().sum()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().rowwise().sum())> == 2);
  static_assert(constant_matrix<decltype(std::declval<Cx1_2>().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx1_2>().rowwise().sum())> == 2);

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
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.colwise().sum()}(), -6);
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.rowwise().sum()}(), -6);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().sum())>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().sum())>);

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
  static_assert(constant_matrix<decltype(std::declval<Cdxx_m2>().abs2().colwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<Cdxx_m2>().abs2().rowwise().sum()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(cd00_22_m2.cwiseAbs2().colwise().sum()), CompileTimeStatus::unknown>);
  EXPECT_EQ(constant_coefficient{cd00_22_m2.cwiseAbs2().colwise().sum()}(), 4);
  EXPECT_EQ(cd00_22_m2.cwiseAbs2().colwise().sum()(0,0), 4);
  EXPECT_EQ(constant_coefficient{Eigen3::EigenWrapper{cxb.array().matrix()}.cwiseAbs2().colwise().sum()}(), 50);
  EXPECT_EQ(Eigen3::EigenWrapper{cxb.array().matrix()}.cwiseAbs2().colwise().sum()(0,0), 50);

  // redux

  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(std::plus<double>{})), CompileTimeStatus::any>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{}))> == -4);

  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})}(), -4);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().redux(std::plus<double>{})}(), -4);

  EXPECT_EQ(c00_22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})(0,0), -4);
  EXPECT_EQ(c00_22_m2.colwise().redux(std::plus<double>{})(0,0), -4);

  // mean -- Note: Eigen version 3.4 calculates x._wise.mean() as if it were x._wise.sum() / dimension.

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  static_assert(constant_matrix<decltype(cd22_m2.colwise().mean()), CompileTimeStatus::unknown>);

  EXPECT_EQ(get_scalar_constant_value(constant_coefficient{(M22::Identity() - M22::Identity()).colwise().mean()}), 0.);
  EXPECT_EQ(get_scalar_constant_value(constant_coefficient{M22::Zero().colwise().mean()}), 0.);
#else
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().mean())> == 2);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().mean())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.rowwise().mean())> == -2);
  static_assert(constant_coefficient_v<decltype(cd20_2_m2.colwise().mean())> == -2);
  static_assert(not constant_matrix<decltype(cd02_2_m2.colwise().mean())>);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().mean())>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().mean())>);
#endif
}


TEST(eigen3, Eigen_PartialReduxExpr_min_max)
{
  auto c22_m2 = (-(M11::Identity() + M11::Identity())).replicate<2,2>();
  auto c00_22_m2 = (-(M11::Identity() + M11::Identity())).replicate(2,2);

  auto cd22_m2 = (-(M22::Identity()+M22::Identity()));
  auto cd20_2_m2 = Eigen::Replicate<decltype(cd22_m2), 1, Eigen::Dynamic> {cd22_m2, 1, 1};
  auto cd02_2_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, 1> {cd22_m2, 1, 1};
  auto cd00_22_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, Eigen::Dynamic> {cd22_m2, 1, 1};

  auto cd22_2 = (M22::Identity() + M22::Identity()).array();
  auto cd20_2_2 = Eigen::Replicate<decltype(cd22_2), 1, Eigen::Dynamic> {cd22_2, 1, 1};
  auto cd02_2_2 = Eigen::Replicate<decltype(cd22_2), Eigen::Dynamic, 1> {cd22_2, 1, 1};
  auto cd00_22_2 = Eigen::Replicate<decltype(cd22_2), Eigen::Dynamic, Eigen::Dynamic> {cd22_2, 1, 1};

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);

  // minCoeff

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C22_2>().rowwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().rowwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const Cx2_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cxx_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const Cxx_2>().rowwise().minCoeff())> == 2);

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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().minCoeff())>);

  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_min_op<double, double>{})), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_min_op<double, double>{}))> == -2);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().redux(Eigen::internal::scalar_min_op<double, double>{})}(), -2);
  EXPECT_EQ(c00_22_m2.colwise().redux(Eigen::internal::scalar_min_op<double, double>{})(0,0), -2);

  // maxCoeff

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cxx_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cxx_2>().rowwise().maxCoeff())> == 2);

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
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().maxCoeff())>);

  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_max_op<double, double>{})), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_max_op<double, double>{}))> == -2);
  EXPECT_EQ(constant_coefficient{c00_22_m2.colwise().redux(Eigen::internal::scalar_max_op<double, double>{})}(), -2);
  EXPECT_EQ(c00_22_m2.colwise().redux(Eigen::internal::scalar_max_op<double, double>{})(0,0), -2);
}


TEST(eigen3, Eigen_PartialReduxExpr_bool)
{
  // all

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_false>().colwise().all())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>().colwise().all())> == false);

  static_assert(constant_matrix<decltype(B22_true::Constant(2).colwise().redux(Eigen::internal::scalar_boolean_and_op{})), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(B22_true::Constant(2).colwise().redux(std::logical_and<bool>{})), CompileTimeStatus::any>);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().redux(Eigen::internal::scalar_boolean_and_op{}))> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().colwise().redux(Eigen::internal::scalar_boolean_and_op{}))> == false);

  // any

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_true>().colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_false>().colwise().any())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>().colwise().any())> == true);

  static_assert(constant_matrix<decltype(B22_true::Constant(2).colwise().redux(Eigen::internal::scalar_boolean_or_op{})), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(B22_true::Constant(2).colwise().redux(std::logical_or<bool>{})), CompileTimeStatus::any>);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().redux(Eigen::internal::scalar_boolean_or_op{}))> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().colwise().redux(Eigen::internal::scalar_boolean_or_op{}))> == false);
}


TEST(eigen3, Eigen_PartialReduxExpr_product)
{
  auto c22_m2 = (-(M11::Identity() + M11::Identity())).replicate<2,2>();
  auto c00_22_m2 = (-(M11::Identity() + M11::Identity())).replicate(2,2);

  auto cd22_m2 = (-(M22::Identity()+M22::Identity()));
  auto cd20_2_m2 = Eigen::Replicate<decltype(cd22_m2), 1, Eigen::Dynamic> {cd22_m2, 1, 1};
  auto cd02_2_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, 1> {cd22_m2, 1, 1};
  auto cd00_22_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, Eigen::Dynamic> {cd22_m2, 1, 1};

  auto cd3322_m2 = cd22_m2.replicate<3,3>();
  auto cd3300_22_m2 = cd00_22_m2.replicate<3,3>();
  auto cd0022_33_m2 = cd22_m2.replicate(3, 3);

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().prod())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().prod())> == 4);
  static_assert(constant_matrix<decltype(std::declval<C2x_2>().rowwise().prod()), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(std::declval<Cx2_2>().colwise().prod()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().prod())> == 4);

  static_assert(zero_matrix<decltype(cd22_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd20_2_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd20_2_m2.rowwise().prod())>);
  static_assert(zero_matrix<decltype(cd02_2_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd02_2_m2.rowwise().prod())>);
  EXPECT_EQ(cd20_2_m2.rowwise().prod()(0,0), 0);
  EXPECT_EQ(cd02_2_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cd00_22_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cd00_22_m2.rowwise().prod()(0,0), 0);

  static_assert(zero_matrix<decltype(cd3322_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd3322_m2.rowwise().prod())>);
  static_assert(zero_matrix<decltype(cd3300_22_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd3300_22_m2.rowwise().prod())>);
  EXPECT_EQ(cd3300_22_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cd3300_22_m2.rowwise().prod()(0,0), 0);
  static_assert(zero_matrix<decltype(cd0022_33_m2.colwise().prod())>);
  static_assert(zero_matrix<decltype(cd0022_33_m2.rowwise().prod())>);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().prod())>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().prod())>);

  // redux

  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_product_op<double, double>{})), CompileTimeStatus::unknown>);
  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(std::multiplies<double>{})), CompileTimeStatus::any>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_product_op<double, double>{}))> == 4);

  EXPECT_EQ(constant_coefficient{c00_22_m2.rowwise().redux(Eigen::internal::scalar_product_op<double, double>{})}(), 4);
  EXPECT_EQ(constant_coefficient{c00_22_m2.rowwise().redux(std::multiplies<double>{})}(), 4);

  EXPECT_EQ(c00_22_m2.rowwise().redux(Eigen::internal::scalar_product_op<double, double>{})(0,0), 4);
  EXPECT_EQ(c00_22_m2.rowwise().redux(std::multiplies<double>{})(0,0), 4);
}


TEST(eigen3, Eigen_PartialReduxExpr_other)
{
  auto cd22_m2 = (-(M22::Identity()+M22::Identity()));
  auto cd20_2_m2 = Eigen::Replicate<decltype(cd22_m2), 1, Eigen::Dynamic> {cd22_m2, 1, 1};
  auto cd02_2_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, 1> {cd22_m2, 1, 1};
  auto cd00_22_m2 = Eigen::Replicate<decltype(cd22_m2), Eigen::Dynamic, Eigen::Dynamic> {cd22_m2, 1, 1};

  auto cd3322_m2 = cd22_m2.replicate<3,3>();
  auto cd3300_22_m2 = cd00_22_m2.replicate<3,3>();
  auto cd0022_33_m2 = cd22_m2.replicate(3, 3);

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);

  // count

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C12_2>().colwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<const C2x_2>().colwise().count())> == 2);
  static_assert(constant_matrix<decltype(std::declval<Cx2_2>().colwise().count()), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(std::declval<Cx2_2>().colwise().count()), CompileTimeStatus::known>);

  static_assert(constant_coefficient_v<decltype(std::declval<const C22_2>().rowwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C21_2>().rowwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().count())> == 2);
  static_assert(constant_matrix<decltype(std::declval<C2x_2>().rowwise().count()), CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<decltype(std::declval<C2x_2>().rowwise().count()), CompileTimeStatus::known>);

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
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.colwise().count()}(), 3);
  EXPECT_EQ(constant_coefficient{cd0022_33_m2.rowwise().count()}(), 3);

  // reverse

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().reverse())> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().colwise().reverse())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().rowwise().reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd22_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd20_2_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd02_2_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd00_22_m2.reverse())> == -2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().reverse())>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().reverse())>);

  // replicate

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate(2))> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(cd22_m2.colwise().replicate<1>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd20_2_m2.colwise().replicate<1>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd02_2_m2.colwise().replicate<1>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd00_22_m2.colwise().replicate<1>())> == -2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().replicate<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().replicate<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().replicate(2))>);
  static_assert(zero_matrix<decltype(std::declval<Zxx>().colwise().replicate(2))>);
}


TEST(eigen3, Eigen_VectorWiseOp)
{
  static_assert(max_indices_of_v<decltype(std::declval<M34>().colwise())> == 2);
  static_assert(std::is_same_v<scalar_type_of_t<decltype(std::declval<M34>().colwise())>, double>);

  static_assert(index_dimension_of_v<decltype(std::declval<M34>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M3x>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M34>().colwise()), 1> == 4);
  static_assert(index_dimension_of_v<decltype(std::declval<Mx4>().colwise()), 1> == 4);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise())> == 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().rowwise())>);
}
