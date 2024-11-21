/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::test;

namespace
{
  auto c11_2 = M11::Identity() + M11::Identity();
  auto cxx_11_2 = Mxx::Identity(1,1) + Mxx::Identity(1,1);

  auto c22_2 = (M11::Identity() + M11::Identity()).replicate<2,2>();
  auto cxx_22_2 = (M11::Identity() + M11::Identity()).replicate(2,2);

  auto c22_m2 = (-(M11::Identity() + M11::Identity())).replicate<2,2>();
  auto cxx_22_m2 = (-(M11::Identity() + M11::Identity())).replicate(2,2);
  auto c22_m2a = -((M11::Identity() + M11::Identity()).replicate<2,2>());
  auto cxx_22_m2a = -((M11::Identity() + M11::Identity()).replicate(2,2));

  auto cd22_2 = M22::Identity() + M22::Identity();
  auto cd2x_2_2 = M2x::Identity(2, 2) + M2x::Identity(2, 2);
  auto cdx2_2_2 = Mx2::Identity(2, 2) + Mx2::Identity(2, 2);
  auto cdxx_22_2 = Mxx::Identity(2, 2) + Mxx::Identity(2, 2);

  auto cd22_m2 = -cd22_2;
  auto cd2x_2_m2 = -cd2x_2_2;
  auto cdx2_2_m2 = -cdx2_2_2;
  auto cdxx_22_m2 = -cdxx_22_2;

  //Replicate(constant_diagonal)
  auto cd3322_2 = cd22_2.replicate<3,3>();
  auto cdxx22_33_2 = cd22_2.replicate(3,3);
  auto cd33xx_22_2 = cdxx_22_2.replicate<3,3>();

  //Replicate(CwiseUnaryOp(constant_diagonal))
  auto cd3322_m2 = cd22_m2.replicate<3,3>();
  auto cd33xx_22_m2 = cdxx_22_m2.replicate<3,3>();
  auto cdxx22_33_m2 = cd22_m2.replicate(3,3);

  //Replicate(Replicate(CwiseUnaryOp(constant_diagonal)))
  auto cd3322_m2a = cd22_m2.replicate<1,1>().replicate<3,3>();
  auto cd33xx_22_m2a = cd22_m2.replicate(1,1).replicate<3,3>();
  auto cdxx22_33_m2a = cd22_m2.replicate(1,1).replicate(3,3);

  //CwiseUnaryOp(Replicate(constant_diagonal))
  auto cd3322_m2b = -cd22_2.replicate<3,3>();
  auto cdxx22_33_m2b = -cd22_2.replicate(3,3);

  //CwiseUnaryOp(CwiseUnaryOp(Replicate(Replicate(CwiseUnaryOp(CwiseUnaryOp(constant diagonal))))))
  auto cd3322_m2c = -(-(-cd22_m2).replicate<1,1>().replicate<3,3>());
  auto cd33xx_22_m2c = (-(-cd22_m2).replicate(1,1)).replicate<3,3>();
  auto cdxx22_33_m2c = (-(-cd22_m2).replicate(1,1)).replicate(3,3);

  auto cxx_21_2 = (M11::Identity() + M11::Identity()).replicate(2,1);

  using P32vert = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Vertical>;
  static_assert(index_dimension_of_v<P32vert, 0> == 1);
  static_assert(index_dimension_of_v<P32vert, 1> == 2);

  using P32horiz = Eigen::PartialReduxExpr<M32, Eigen::internal::member_sum<double, double>, Eigen::Horizontal>;
  static_assert(index_dimension_of_v<P32horiz, 0> == 3);
  static_assert(index_dimension_of_v<P32horiz, 1> == 1);
}


TEST(eigen3, Eigen_PartialReduxExpr_lpNorm0)
{
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cxx_22_m2.colwise().lpNorm<0>())>>);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cxx_22_m2.colwise().lpNorm<0>()(0,0), INFINITY);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<0>()(0,0), INFINITY);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd2x_2_m2.rowwise().lpNorm<0>())>>);
  EXPECT_EQ(constant_coefficient{cd2x_2_m2.rowwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cd2x_2_m2.rowwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdx2_2_m2.colwise().lpNorm<0>())>>);
  EXPECT_EQ(constant_coefficient{cdx2_2_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cdx2_2_m2.colwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx_22_m2.colwise().lpNorm<0>())>>);
  EXPECT_EQ(constant_coefficient{cdxx_22_m2.colwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cdxx_22_m2.colwise().lpNorm<0>()(0,0), INFINITY);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx_22_m2.rowwise().lpNorm<0>())>>);
  EXPECT_EQ(constant_coefficient{cdxx_22_m2.rowwise().lpNorm<0>()}(), INFINITY);
  EXPECT_EQ(cdxx_22_m2.rowwise().lpNorm<0>()(0,0), INFINITY);

  EXPECT_EQ(M22::Zero().colwise().lpNorm<0>()(0,0), INFINITY);
}


TEST(eigen3, Eigen_PartialReduxExpr_lpNorm1)
{
  static_assert(value::static_scalar<constant_coefficient<decltype(c22_m2)>>);
  static_assert(not zero<decltype(c22_m2)>);
  static_assert(not one_dimensional<decltype(c22_m2), Qualification::depends_on_dynamic_shape>);
  static_assert(not constant_diagonal_matrix<decltype(c22_m2)>);

  //constant
  static_assert(constant_coefficient_v<decltype(c11_2.colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(c11_2.rowwise().lpNorm<1>())> == 2);
  EXPECT_EQ(constant_coefficient{c11_2.colwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(c11_2.colwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{c11_2.rowwise().lpNorm<1>()}(), 2);
  EXPECT_EQ(c11_2.rowwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{M22::Constant(2).colwise().lpNorm<1>()}(), 4);
  static_assert(zero<decltype(std::declval<Z22>().colwise().lpNorm<1>())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().lpNorm<1>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<1>()(0,0), 0);

  //Replicate(constant)
  static_assert(constant_coefficient_v<decltype(c22_2.colwise().lpNorm<1>())> == 4);
  static_assert(constant_coefficient_v<decltype(c22_2.rowwise().lpNorm<1>())> == 4);
  EXPECT_EQ(constant_coefficient{c22_2.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{c22_2.rowwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_2.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_2.rowwise().lpNorm<1>()}(), 4);

  //CwiseUnaryOp(constant)
  static_assert(constant_coefficient_v<decltype((-c11_2).colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype((-c11_2).rowwise().lpNorm<1>())> == 2);
  EXPECT_EQ(constant_coefficient{(-c11_2).colwise().lpNorm<1>()}(), 2);
  EXPECT_EQ((-c11_2).colwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{(-c11_2).rowwise().lpNorm<1>()}(), 2);
  EXPECT_EQ((-c11_2).rowwise().lpNorm<1>()(0,0), 2);

  //CwiseUnaryOp(CwiseUnaryOp(constant))
  static_assert(constant_coefficient_v<decltype((-(-c11_2)).colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype((-(-c11_2)).rowwise().lpNorm<1>())> == 2);
  EXPECT_EQ(constant_coefficient{(-(-c11_2)).colwise().lpNorm<1>()}(), 2);
  EXPECT_EQ((-(-c11_2)).colwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{(-(-c11_2)).rowwise().lpNorm<1>()}(), 2);
  EXPECT_EQ((-(-c11_2)).rowwise().lpNorm<1>()(0,0), 2);

  //Replicate(CwiseUnaryOp(constant))
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<1>())> == 4);
  static_assert(constant_coefficient_v<decltype(c22_m2.rowwise().lpNorm<1>())> == 4);
  EXPECT_EQ(constant_coefficient{c22_m2.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{c22_m2.rowwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(cxx_22_m2.colwise().lpNorm<1>()(0,0), 4);
  EXPECT_EQ(cxx_22_m2.rowwise().lpNorm<1>()(0,0), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.rowwise().lpNorm<1>()}(), 4);

  //CwiseUnaryOp(Replicate(constant))
  static_assert(constant_coefficient_v<decltype(c22_m2a.colwise().lpNorm<1>())> == 4);
  static_assert(constant_coefficient_v<decltype(c22_m2a.rowwise().lpNorm<1>())> == 4);
  EXPECT_EQ(constant_coefficient{c22_m2a.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{c22_m2a.rowwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(cxx_22_m2a.colwise().lpNorm<1>()(0,0), 4);
  EXPECT_EQ(cxx_22_m2a.rowwise().lpNorm<1>()(0,0), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_m2a.colwise().lpNorm<1>()}(), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_m2a.rowwise().lpNorm<1>()}(), 4);

  //constant_diagonal
  static_assert(constant_coefficient_v<decltype(cd22_2.colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_2.rowwise().lpNorm<1>())> == 2);
  static_assert(not constant_matrix<decltype(cdxx_22_2.colwise().lpNorm<1>())>);
  static_assert(not constant_matrix<decltype(cdxx_22_2.rowwise().lpNorm<1>())>);
  EXPECT_EQ(cdxx_22_2.colwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(cdxx_22_2.rowwise().lpNorm<1>()(0,0), 2);

  //Replicate(constant_diagonal)
  static_assert(constant_coefficient_v<decltype(cd3322_2.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3322_2.rowwise().lpNorm<1>())> == 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_2.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_2.rowwise().lpNorm<1>()}(), 6);
  static_assert(not constant_matrix<decltype(cd33xx_22_2.colwise().lpNorm<1>())>);
  static_assert(not constant_matrix<decltype(cd33xx_22_2.rowwise().lpNorm<1>())>);
  EXPECT_EQ((cd3322_2.colwise().lpNorm<1>()(0,0)), 6);
  EXPECT_EQ((cd3322_2.rowwise().lpNorm<1>()(0,0)), 6);

  //CwiseUnaryOp(constant_diagonal)
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().lpNorm<1>())> == 2);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().lpNorm<1>())>);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.rowwise().lpNorm<1>())>);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<1>()(0,0), 2);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<1>()(0,0), 2);

  //Replicate(CwiseUnaryOp(constant_diagonal))
  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().lpNorm<1>())> == 6);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.colwise().lpNorm<1>())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.rowwise().lpNorm<1>())>>);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.rowwise().lpNorm<1>()}(), 6);

  //Replicate(Replicate(CwiseUnaryOp(constant_diagonal)))
  static_assert(constant_coefficient_v<decltype(cd3322_m2a.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2a.rowwise().lpNorm<1>())> == 6);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2a.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2a.rowwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2a.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2a.rowwise().lpNorm<1>()}(), 6);

  //CwiseUnaryOp(Replicate(constant_diagonal))
  static_assert(constant_coefficient_v<decltype(cd3322_m2b.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2b.rowwise().lpNorm<1>())> == 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2b.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2b.rowwise().lpNorm<1>()}(), 6);

  //CwiseUnaryOp(CwiseUnaryOp(Replicate(Replicate(CwiseUnaryOp(CwiseUnaryOp(constant diagonal))))))
  static_assert(constant_coefficient_v<decltype(cd3322_m2c.colwise().lpNorm<1>())> == 6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2c.rowwise().lpNorm<1>())> == 6);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2c.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2c.rowwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2c.colwise().lpNorm<1>()}(), 6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2c.rowwise().lpNorm<1>()}(), 6);
}


TEST(eigen3, Eigen_PartialReduxExpr_lpNorm2)
{
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<2>())>, internal::constexpr_sqrt(8.)));
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(c22_m2a.rowwise().lpNorm<2>())>, internal::constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{c22_m2.colwise().lpNorm<2>()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(constant_coefficient{c22_m2a.rowwise().lpNorm<2>()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(constant_coefficient{cxx_22_m2.colwise().lpNorm<2>()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(constant_coefficient{cxx_22_m2a.rowwise().lpNorm<2>()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(cxx_22_m2.colwise().lpNorm<2>()(0,0), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(cxx_22_m2a.rowwise().lpNorm<2>()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<2>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<2>()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<2>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().lpNorm<2>())> == 2);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().lpNorm<2>())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().lpNorm<2>())> == 2 * numbers::sqrt3);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.colwise().lpNorm<2>())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.rowwise().lpNorm<2>())>>);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.colwise().lpNorm<2>()}, 2 * numbers::sqrt3);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.rowwise().lpNorm<2>()}, 2 * numbers::sqrt3);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.colwise().lpNorm<2>())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.rowwise().lpNorm<2>())>>);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2a.colwise().lpNorm<2>()}, 2 * numbers::sqrt3);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2a.rowwise().lpNorm<2>()}, 2 * numbers::sqrt3);

  static_assert(zero<decltype(std::declval<Z22>().colwise().lpNorm<2>())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().lpNorm<2>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<2>()(0,0), 0);
}


TEST(eigen3, Eigen_PartialReduxExpr_lpNorm3)
{
  static_assert(internal::are_within_tolerance<5>(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<3>())>, internal::constexpr_pow(16., 1./3)));
  EXPECT_NEAR(constant_coefficient{c22_m2.colwise().lpNorm<3>()}(), (std::pow(16., 1./3)), 1e-9);
  EXPECT_NEAR(constant_coefficient{cxx_22_m2.colwise().lpNorm<3>()}(), (std::pow(16., 1./3)), 1e-9);
  EXPECT_NEAR(cxx_22_m2.colwise().lpNorm<3>()(0,0), (std::pow(16., 1./3)), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<3>()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().lpNorm<3>()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().lpNorm<3>()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<3>())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().lpNorm<3>())> == 2);

  static_assert(internal::are_within_tolerance<5>(constant_coefficient_v<decltype(cd3322_m2.colwise().lpNorm<3>())>, 2.88449914061481676464327662)); // cube root of 24
  static_assert(internal::are_within_tolerance<5>(constant_coefficient_v<decltype(cd3322_m2.rowwise().lpNorm<3>())>, 2.88449914061481676464327662));
  EXPECT_NEAR(constant_coefficient{cdxx22_33_m2.colwise().lpNorm<3>()}, 2.88449914061481676464327662, 1e-12);
  EXPECT_NEAR(constant_coefficient{cdxx22_33_m2.rowwise().lpNorm<3>()}, 2.88449914061481676464327662, 1e-12);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.colwise().lpNorm<3>())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.rowwise().lpNorm<3>())>>);
  EXPECT_NEAR(constant_coefficient{cd33xx_22_m2a.colwise().lpNorm<3>()}, 2.88449914061481676464327662, 1e-12);
  EXPECT_NEAR(constant_coefficient{cd33xx_22_m2a.rowwise().lpNorm<3>()}, 2.88449914061481676464327662, 1e-12);

  static_assert(zero<decltype(std::declval<Z22>().colwise().lpNorm<3>())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().lpNorm<3>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<3>()(0,0), 0);
}


TEST(eigen3, Eigen_PartialReduxExpr_lpNorm_infinity)
{
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().lpNorm<Eigen::Infinity>())> == 2);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().lpNorm<Eigen::Infinity>()}(), 2);
  EXPECT_EQ(cxx_22_m2.colwise().lpNorm<Eigen::Infinity>()(0,0), 2);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().lpNorm<Eigen::Infinity>())> == 2);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().lpNorm<Eigen::Infinity>())>);
  EXPECT_EQ(cd22_m2.colwise().lpNorm<Eigen::Infinity>()(0,0), 2);

  static_assert(zero<decltype(std::declval<Z22>().colwise().lpNorm<Eigen::Infinity>())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().lpNorm<Eigen::Infinity>())>);
  EXPECT_EQ(M22::Zero().colwise().lpNorm<Eigen::Infinity>()(0,0), 0);
}


TEST(eigen3, Eigen_PartialReduxExpr_stableNorm)
{
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().stableNorm())>, internal::constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{cxx_22_m2.colwise().stableNorm()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(cxx_22_m2.colwise().stableNorm()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().stableNorm()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().stableNorm()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().stableNorm()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().stableNorm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().stableNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().stableNorm())> == 2);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().stableNorm())>);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.rowwise().stableNorm())>);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().stableNorm())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().stableNorm())> == 2 * numbers::sqrt3);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.colwise().stableNorm())>);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.rowwise().stableNorm())>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.colwise().stableNorm())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.rowwise().stableNorm())>>);
  EXPECT_NEAR(constant_coefficient{cdxx22_33_m2.colwise().stableNorm()}, 2 * numbers::sqrt3, 1e-9);
  EXPECT_NEAR(constant_coefficient{cdxx22_33_m2.rowwise().stableNorm()}, 2 * numbers::sqrt3, 1e-9);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.colwise().stableNorm())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.rowwise().stableNorm())>>);
  EXPECT_NEAR(constant_coefficient{cd33xx_22_m2a.colwise().stableNorm()}, 2 * numbers::sqrt3, 1e-9);
  EXPECT_NEAR(constant_coefficient{cd33xx_22_m2a.rowwise().stableNorm()}, 2 * numbers::sqrt3, 1e-9);
  EXPECT_DOUBLE_EQ(cd33xx_22_m2a.rowwise().stableNorm()(0,0), 2 * numbers::sqrt3);

  static_assert(zero<decltype(std::declval<Z22>().colwise().stableNorm())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().stableNorm())>);
}


TEST(eigen3, Eigen_PartialReduxExpr_hypotNorm)
{
  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().hypotNorm())>, internal::constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{cxx_22_m2.colwise().hypotNorm()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(cxx_22_m2.colwise().hypotNorm()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().hypotNorm()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().hypotNorm()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().hypotNorm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().hypotNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().hypotNorm())> == 2);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().hypotNorm())>);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.rowwise().hypotNorm())>);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().hypotNorm())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().hypotNorm())> == 2 * numbers::sqrt3);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.colwise().hypotNorm())>);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.rowwise().hypotNorm())>);
  EXPECT_NEAR(constant_coefficient {cd33xx_22_m2a.colwise().hypotNorm()}, 2 * numbers::sqrt3, 1e-9);
  EXPECT_NEAR(constant_coefficient {cd33xx_22_m2a.rowwise().hypotNorm()}, 2 * numbers::sqrt3, 1e-9);
  EXPECT_DOUBLE_EQ(cd33xx_22_m2a.rowwise().hypotNorm()(0,0), 2 * numbers::sqrt3);

  static_assert(zero<decltype(std::declval<Z22>().colwise().hypotNorm())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().hypotNorm())>);
}


TEST(eigen3, Eigen_PartialReduxExpr_squaredNorm)
{
  // Note: Eigen version 3.4 calculates x._wise().squaredNorm() as if it were x.cwiseAbs2()._wise().sum().

  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().squaredNorm())> == 8);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().squaredNorm()}(), 8);
  EXPECT_EQ(cxx_22_m2.colwise().squaredNorm()(0,0), 8);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd22_m2.colwise().squaredNorm()(0,0), 4);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().squaredNorm()}(), 4);
  EXPECT_EQ(cd22_m2.rowwise().squaredNorm()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().squaredNorm())> == 4);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().squaredNorm())> == 4);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().squaredNorm())>);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.rowwise().squaredNorm())>);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().squaredNorm())> == 12);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().squaredNorm())> == 12);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.colwise().squaredNorm())>);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.rowwise().squaredNorm())>);
  EXPECT_NEAR(constant_coefficient {cd33xx_22_m2a.colwise().squaredNorm()}, 12, 1e-9);
  EXPECT_NEAR(constant_coefficient {cd33xx_22_m2a.rowwise().squaredNorm()}, 12, 1e-9);

  static_assert(zero<decltype(std::declval<Z22>().colwise().squaredNorm())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().squaredNorm())>);
}


TEST(eigen3, Eigen_PartialReduxExpr_norm)
{
  // Note: Eigen version 3.4 calculates x._wise().norm() as if it were x.cwiseAbs2()._wise().sum().sqrt().

  static_assert(internal::are_within_tolerance(constant_coefficient_v<decltype(c22_m2.colwise().norm())>, internal::constexpr_sqrt(8.)));
  EXPECT_NEAR(constant_coefficient{cxx_22_m2.colwise().norm()}(), std::sqrt(8.), 1e-9);
  EXPECT_NEAR(cxx_22_m2.colwise().norm()(0,0), std::sqrt(8.), 1e-9);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().norm()}(), 2);
  EXPECT_EQ(cd22_m2.colwise().norm()(0,0), 2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().norm()}(), 2);
  EXPECT_EQ(cd22_m2.rowwise().norm()(0,0), 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().norm())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().norm())> == 2);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().norm())>);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.rowwise().norm())>);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().norm())> == 2 * numbers::sqrt3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().norm())> == 2 * numbers::sqrt3);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.colwise().norm())>);
  static_assert(not constant_matrix<decltype(cd33xx_22_m2.rowwise().norm())>);
  EXPECT_NEAR(constant_coefficient {cd33xx_22_m2a.colwise().norm()}, 2 * numbers::sqrt3, 1e-9);
  EXPECT_NEAR(constant_coefficient {cd33xx_22_m2a.rowwise().norm()}, 2 * numbers::sqrt3, 1e-9);

  static_assert(zero<decltype(std::declval<Z22>().colwise().norm())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().norm())>);
}


TEST(eigen3, Eigen_PartialReduxExpr_sum)
{
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex

  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().sum()}(), -4);
  EXPECT_EQ(cxx_22_m2.colwise().sum()(0,0), -4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().sum())> == 4);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<C2x_2>().rowwise().sum())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<Cx2_2>().colwise().sum())>>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().sum())> == 4);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<Cxx_2>().colwise().sum())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<Cxx_2>().rowwise().sum())>>);

  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_m2>().colwise().sum())> == -2);
  static_assert(not constant_matrix<decltype(std::declval<Cdxx_m2>().colwise().sum())>);

  EXPECT_EQ(constant_coefficient{cxx_21_2.replicate(2,2).colwise().sum()}(), 8);
  EXPECT_EQ(cxx_21_2.replicate(2,2).colwise().sum()(0,0), 8);
  EXPECT_EQ(constant_coefficient{cxx_21_2.replicate(2,2).rowwise().sum()}(), 4);
  EXPECT_EQ(cxx_21_2.replicate(2,2).rowwise().sum()(0,0), 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().rowwise().sum())> == 2);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<Cx1_2>().colwise().sum())>>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx1_2>().rowwise().sum())> == 2);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().sum()}(), -2);
  EXPECT_EQ(cd22_m2.colwise().sum()(0,0), -2);
  EXPECT_EQ(constant_coefficient{cd22_m2.rowwise().sum()}(), -2);
  EXPECT_EQ(cd22_m2.rowwise().sum()(0,0), -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().sum())> == -2);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.colwise().sum())>);
  static_assert(not constant_matrix<decltype(cdxx_22_m2.rowwise().sum())>);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().sum())> == -6);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().sum())> == -6);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.colwise().sum())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cdxx22_33_m2.rowwise().sum())>>);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.colwise().sum()}(), -6);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.rowwise().sum()}(), -6);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.colwise().sum())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd33xx_22_m2a.rowwise().sum())>>);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2a.colwise().sum()}(), -6);
  EXPECT_EQ(constant_coefficient{cd33xx_22_m2a.rowwise().sum()}(), -6);

  static_assert(zero<decltype(std::declval<Z22>().colwise().sum())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().sum())>);

  EXPECT_EQ(constant_coefficient{cxb.colwise().sum().imag()}(), 8);
  EXPECT_EQ(constant_coefficient{cxb.rowwise().sum().imag()}(), 8);
  EXPECT_EQ(cxb.colwise().sum().imag()(0,0), 8);
  EXPECT_EQ(cxb.rowwise().sum().imag()(0,0), 8);
  EXPECT_EQ(constant_coefficient{Eigen3::make_eigen_wrapper(cxb.array().matrix()).colwise().sum().imag()}(), 8);
  EXPECT_EQ(constant_coefficient{Eigen3::make_eigen_wrapper(cxb.array().matrix()).rowwise().sum().imag()}(), 8);
  EXPECT_EQ(cxb.array().matrix().colwise().sum().imag()(0,0), 8);
  EXPECT_EQ(cxb.array().matrix().rowwise().sum().imag()(0,0), 8);
  EXPECT_EQ(Eigen3::make_eigen_wrapper(cxb.array().matrix()).colwise().sum().imag()(0,0), 8);
  EXPECT_EQ(Eigen3::make_eigen_wrapper(cxb.array().matrix()).rowwise().sum().imag()(0,0), 8);

  EXPECT_EQ(constant_coefficient{Eigen3::make_eigen_wrapper(cxb.array().matrix()).imag().colwise().sum()}(), 8);
  EXPECT_EQ(constant_coefficient{Eigen3::make_eigen_wrapper(cxb.array().matrix()).imag().rowwise().sum()}(), 8);
  EXPECT_EQ(cxb.array().matrix().imag().colwise().sum()(0,0), 8);
  EXPECT_EQ(cxb.array().matrix().imag().rowwise().sum()(0,0), 8);
  EXPECT_EQ(Eigen3::make_eigen_wrapper(cxb.array().matrix()).imag().colwise().sum()(0,0), 8);
  EXPECT_EQ(Eigen3::make_eigen_wrapper(cxb.array().matrix()).imag().rowwise().sum()(0,0), 8);

  static_assert(constant_coefficient_v<decltype(c22_m2.cwiseAbs2().colwise().sum())> == 8);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cxx_22_m2.cwiseAbs2().rowwise().sum())>>);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.cwiseAbs2().colwise().sum()}(), 8);

  static_assert(constant_coefficient_v<decltype(cd22_m2.cwiseAbs2().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(Eigen3::make_eigen_wrapper(cd22_m2.array().matrix()).cwiseAbs2().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_m2>().abs2().colwise().sum())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_m2>().abs2().rowwise().sum())> == 4);
  EXPECT_EQ(cd22_m2.cwiseAbs2().colwise().sum()(0,0), 4);
  EXPECT_EQ(constant_coefficient{Eigen3::make_eigen_wrapper(cxb.array().matrix()).cwiseAbs2().colwise().sum()}(), 50);
  EXPECT_EQ(Eigen3::make_eigen_wrapper(cxb.array().matrix()).cwiseAbs2().colwise().sum()(0,0), 50);

  // redux

  static_assert(value::dynamic_scalar<constant_coefficient<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_sum_op<double, double>{}))>>);
  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(std::plus<double>{}))>>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{}))> == -4);

  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})}(), -4);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().redux(std::plus<double>{})}(), -4);

  EXPECT_EQ(cxx_22_m2.colwise().redux(Eigen::internal::scalar_sum_op<double, double>{})(0,0), -4);
  EXPECT_EQ(cxx_22_m2.colwise().redux(std::plus<double>{})(0,0), -4);

  // mean -- Note: Eigen version 3.4 calculates x._wise.mean() as if it were x._wise.sum() / dimension.

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(cd22_m2.colwise().mean())>>);

  EXPECT_EQ(value::to_number(constant_coefficient{(M22::Identity() - M22::Identity()).colwise().mean()}), 0.);
  EXPECT_EQ(value::to_number(constant_coefficient{M22::Zero().colwise().mean()}), 0.);
#else
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().mean())> == 2);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().mean())> == -2);
  static_assert(constant_coefficient_v<decltype(cd2x_2_m2.rowwise().mean())> == -2);
  static_assert(constant_coefficient_v<decltype(cd2x_2_m2.colwise().mean())> == -2);
  static_assert(not constant_matrix<decltype(cdx2_2_m2.colwise().mean())>);

  static_assert(zero<decltype(std::declval<Z22>().colwise().mean())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().mean())>);
#endif
}


TEST(eigen3, Eigen_PartialReduxExpr_minCoeff)
{
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
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd22_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd22_2.rowwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd2x_2_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd2x_2_2.rowwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdx2_2_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdx2_2_2.rowwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdxx_22_2.colwise().minCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdxx_22_2.rowwise().minCoeff())> == 0);
  EXPECT_EQ(constant_coefficient{cd2x_2_2.rowwise().minCoeff()}(), 0);
  EXPECT_EQ(cd2x_2_2.rowwise().minCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cdx2_2_2.colwise().minCoeff()}(), 0);
  EXPECT_EQ(cdx2_2_2.colwise().minCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cdxx_22_2.rowwise().minCoeff()}(), 0);
  EXPECT_EQ(cdxx_22_2.rowwise().minCoeff()(0,0), 0);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().minCoeff()}(), -2);
  EXPECT_EQ(cd22_m2.colwise().minCoeff()(0,0), -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().minCoeff())> == -2);
  EXPECT_EQ(cd22_m2.rowwise().minCoeff()(0,0), -2);
  EXPECT_EQ(cd22_m2.colwise().minCoeff()(0,0), -2);

  static_assert(constant_coefficient_v<decltype(cd3322_m2a.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd3322_m2a.rowwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd33xx_22_m2a.colwise().minCoeff())> == -2);
  static_assert(constant_coefficient_v<decltype(cd33xx_22_m2a.rowwise().minCoeff())> == -2);

  static_assert(zero<decltype(std::declval<Z22>().colwise().minCoeff())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().minCoeff())>);

  static_assert(value::dynamic_scalar<constant_coefficient<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_min_op<double, double>{}))>>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_min_op<double, double>{}))> == -2);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().redux(Eigen::internal::scalar_min_op<double, double>{})}(), -2);
  EXPECT_EQ(cxx_22_m2.colwise().redux(Eigen::internal::scalar_min_op<double, double>{})(0,0), -2);
}


TEST(eigen3, Eigen_PartialReduxExpr_maxCoeff)
{
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
  static_assert(constant_coefficient_v<decltype(cd22_2.rowwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd2x_2_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd2x_2_m2.rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdx2_2_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdx2_2_m2.rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdxx_22_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdxx_22_m2.rowwise().maxCoeff())> == 0);
  EXPECT_EQ(constant_coefficient{cd2x_2_m2.rowwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd2x_2_m2.rowwise().maxCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cdx2_2_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cdx2_2_m2.colwise().maxCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cdxx_22_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cdxx_22_m2.colwise().maxCoeff()(0,0), 0);

  EXPECT_EQ(constant_coefficient{cd22_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd22_m2.colwise().maxCoeff()(0,0), 0);
  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd2x_2_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd2x_2_m2.rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdx2_2_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdx2_2_m2.rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdxx_22_m2.colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cdxx_22_m2.rowwise().maxCoeff())> == 0);
  EXPECT_EQ(constant_coefficient{cd2x_2_m2.rowwise().maxCoeff()}(), 0);
  EXPECT_EQ(cd2x_2_m2.rowwise().maxCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cdx2_2_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cdx2_2_m2.colwise().maxCoeff()(0,0), 0);
  EXPECT_EQ(constant_coefficient{cdxx_22_m2.colwise().maxCoeff()}(), 0);
  EXPECT_EQ(cdxx_22_m2.colwise().maxCoeff()(0,0), 0);

  static_assert(constant_coefficient_v<decltype(cd22_m2.replicate<1,1>().replicate<3,3>().colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd22_m2.replicate<1,1>().replicate<3,3>().rowwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd22_m2.replicate(1,1).replicate<3,3>().colwise().maxCoeff())> == 0);
  static_assert(constant_coefficient_v<decltype(cd22_m2.replicate(1,1).replicate<3,3>().rowwise().maxCoeff())> == 0);

  static_assert(zero<decltype(std::declval<Z22>().colwise().maxCoeff())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().maxCoeff())>);

  static_assert(value::dynamic_scalar<constant_coefficient<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_max_op<double, double>{}))>>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_max_op<double, double>{}))> == -2);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.colwise().redux(Eigen::internal::scalar_max_op<double, double>{})}(), -2);
  EXPECT_EQ(cxx_22_m2.colwise().redux(Eigen::internal::scalar_max_op<double, double>{})(0,0), -2);
}


TEST(eigen3, Eigen_PartialReduxExpr_all)
{
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_false>().colwise().all())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>().colwise().all())> == false);

  static_assert(value::dynamic_scalar<constant_coefficient<decltype(B22_true::Constant(2).colwise().redux(Eigen::internal::scalar_boolean_and_op{}))>>);
  static_assert(constant_matrix<decltype(B22_true::Constant(2).colwise().redux(std::logical_and<bool>{}))>>);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().redux(Eigen::internal::scalar_boolean_and_op{}))> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().colwise().redux(Eigen::internal::scalar_boolean_and_op{}))> == false);
}


TEST(eigen3, Eigen_PartialReduxExpr_any)
{
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_true>().colwise().any())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B11_false>().colwise().any())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>().colwise().any())> == true);

  static_assert(value::dynamic_scalar<constant_coefficient<decltype(B22_true::Constant(2).colwise().redux(Eigen::internal::scalar_boolean_or_op{}))>>);
  static_assert(constant_matrix<decltype(B22_true::Constant(2).colwise().redux(std::logical_or<bool>{}))>>);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().redux(Eigen::internal::scalar_boolean_or_op{}))> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().colwise().redux(Eigen::internal::scalar_boolean_or_op{}))> == false);
}


TEST(eigen3, Eigen_PartialReduxExpr_product)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().prod())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().colwise().prod())> == 4);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<C2x_2>().rowwise().prod())>>);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<Cx2_2>().colwise().prod())>>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().prod())> == 4);

  static_assert(zero<decltype(cd22_m2.colwise().prod())>);
  static_assert(zero<decltype(cd2x_2_m2.colwise().prod())>);
  static_assert(zero<decltype(cd2x_2_m2.rowwise().prod())>);
  static_assert(zero<decltype(cdx2_2_m2.colwise().prod())>);
  static_assert(zero<decltype(cdx2_2_m2.rowwise().prod())>);
  EXPECT_EQ(cd2x_2_m2.rowwise().prod()(0,0), 0);
  EXPECT_EQ(cdx2_2_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cdxx_22_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cdxx_22_m2.rowwise().prod()(0,0), 0);

  static_assert(zero<decltype(cd3322_m2.colwise().prod())>);
  static_assert(zero<decltype(cd3322_m2.rowwise().prod())>);
  static_assert(zero<decltype(cd33xx_22_m2.colwise().prod())>);
  static_assert(zero<decltype(cd33xx_22_m2.rowwise().prod())>);
  EXPECT_EQ(cd33xx_22_m2.colwise().prod()(0,0), 0);
  EXPECT_EQ(cd33xx_22_m2.rowwise().prod()(0,0), 0);
  static_assert(zero<decltype(cdxx22_33_m2.colwise().prod())>);
  static_assert(zero<decltype(cdxx22_33_m2.rowwise().prod())>);

  static_assert(zero<decltype(std::declval<Z22>().colwise().prod())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().prod())>);

  // redux

  static_assert(value::dynamic_scalar<constant_coefficient<decltype(M22::Constant(2).colwise().redux(Eigen::internal::scalar_product_op<double, double>{}))>>);
  static_assert(constant_matrix<decltype(M22::Constant(2).colwise().redux(std::multiplies<double>{}))>>);
  static_assert(constant_coefficient_v<decltype(c22_m2.colwise().redux(Eigen::internal::scalar_product_op<double, double>{}))> == 4);

  EXPECT_EQ(constant_coefficient{cxx_22_m2.rowwise().redux(Eigen::internal::scalar_product_op<double, double>{})}(), 4);
  EXPECT_EQ(constant_coefficient{cxx_22_m2.rowwise().redux(std::multiplies<double>{})}(), 4);

  EXPECT_EQ(cxx_22_m2.rowwise().redux(Eigen::internal::scalar_product_op<double, double>{})(0,0), 4);
  EXPECT_EQ(cxx_22_m2.rowwise().redux(std::multiplies<double>{})(0,0), 4);
}


TEST(eigen3, Eigen_PartialReduxExpr_count)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C12_2>().colwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<const C2x_2>().colwise().count())> == 2);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<Cx2_2>().colwise().count())>>);
  static_assert(not value::static_scalar<constant_coefficient<decltype(std::declval<Cx2_2>().colwise().count())>>);

  static_assert(constant_coefficient_v<decltype(std::declval<const C22_2>().rowwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<const C21_2>().rowwise().count())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<Cx2_2>().rowwise().count())> == 2);
  static_assert(value::dynamic_scalar<constant_coefficient<decltype(std::declval<C2x_2>().rowwise().count())>>);
  static_assert(not value::static_scalar<constant_coefficient<decltype(std::declval<C2x_2>().rowwise().count())>>);

  static_assert(constant_coefficient_v<decltype(cd22_m2.colwise().count())> == 1);
  EXPECT_EQ(cdx2_2_m2.colwise().count()(0,0), 1);

  static_assert(constant_coefficient_v<decltype(cd22_m2.rowwise().count())> == 1);
  EXPECT_EQ(cd2x_2_m2.rowwise().count()(0,0), 1);

  static_assert(constant_coefficient_v<decltype(cd3322_m2.colwise().count())> == 3);
  static_assert(constant_coefficient_v<decltype(cd3322_m2.rowwise().count())> == 3);
  EXPECT_NEAR(constant_coefficient {cdxx22_33_m2.colwise().count()}, 3, 1e-9);
  EXPECT_NEAR(constant_coefficient {cdxx22_33_m2.rowwise().count()}, 3, 1e-9);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.colwise().count()}(), 3);
  EXPECT_EQ(constant_coefficient{cdxx22_33_m2.rowwise().count()}(), 3);
}


TEST(eigen3, Eigen_PartialReduxExpr_reverse)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().reverse())> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().colwise().reverse())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().rowwise().reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd22_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd2x_2_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cdx2_2_m2.reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cdxx_22_m2.reverse())> == -2);

  static_assert(zero<decltype(std::declval<Z22>().colwise().reverse())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().reverse())>);
}


TEST(eigen3, Eigen_PartialReduxExpr_replicate)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate(2))> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(cd22_m2.colwise().replicate<1>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cd2x_2_m2.colwise().replicate<1>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cdx2_2_m2.colwise().replicate<1>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(cdxx_22_m2.colwise().replicate<1>())> == -2);

  static_assert(zero<decltype(std::declval<Z22>().colwise().replicate<2>())>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().replicate<2>())>);
  static_assert(zero<decltype(std::declval<Z22>().colwise().replicate(2))>);
  static_assert(zero<decltype(std::declval<Zxx>().colwise().replicate(2))>);
}

