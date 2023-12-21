/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
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


TEST(eigen3, constants)
{
  static_assert(not constant_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);
  static_assert(constant_diagonal_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);

  static_assert(constant_matrix<Z21, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(scalar_constant<constant_coefficient<Z21>, CompileTimeStatus::known>);
  static_assert(not scalar_constant<constant_coefficient<Z21>, CompileTimeStatus::unknown>);

  static_assert(constant_matrix<M11, CompileTimeStatus::unknown>);
  static_assert(constant_matrix<M1x, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(constant_matrix<Mx1, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(constant_matrix<Mxx, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(not constant_matrix<M21, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_matrix<M21, CompileTimeStatus::unknown, Likelihood::maybe>);
  EXPECT_EQ(constant_coefficient{make_dense_object_from<M11>(5.5)}(), 5.5);
  static_assert(constant_diagonal_matrix<Mxx, CompileTimeStatus::unknown, Likelihood::maybe>);
  EXPECT_EQ(constant_diagonal_coefficient{make_dense_object_from<M11>(5.5)}(), 5.5);

  static_assert(constant_coefficient_v<Z21> == 0);
  static_assert(constant_coefficient_v<Z12> == 0);
  static_assert(constant_coefficient_v<Z23> == 0);
  static_assert(constant_coefficient_v<Z2x> == 0);
  static_assert(constant_coefficient_v<Zx2> == 0);

  static_assert(constant_coefficient_v<Zxx> == 0);
  static_assert(constant_coefficient_v<Zx1> == 0);
  static_assert(constant_coefficient_v<C11_1> == 1);
  static_assert(constant_coefficient_v<C11_m1> == -1);
  static_assert(constant_coefficient_v<C11_2> == 2);
  static_assert(constant_coefficient_v<C11_m2> == -2);
  static_assert(constant_coefficient_v<C11_3> == 3);
  static_assert(constant_coefficient_v<C2x_2> == 2);
  static_assert(constant_coefficient_v<Cx2_2> == 2);
  static_assert(constant_coefficient_v<Cxx_2> == 2);
  static_assert(constant_coefficient_v<B22_true> == true);
  static_assert(constant_coefficient_v<B22_false> == false);
  static_assert(not constant_matrix<Cd22_2>);
  static_assert(not constant_matrix<Cd2x_2>);
  static_assert(not constant_matrix<Cdx2_2>);
  static_assert(not constant_matrix<Cd22_3>);
  static_assert(not constant_matrix<Cd22_2, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd2x_2, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<Cdx2_2, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd22_3, CompileTimeStatus::any, Likelihood::maybe>);

  constant_coefficient<C21_3> c3;
  static_assert(std::decay_t<decltype(+c3)>::value == 3);
  static_assert(std::decay_t<decltype(-c3)>::value == -3);
  static_assert(std::decay_t<decltype(c3 + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(c3 - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(c3 * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(c3 / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + c3 == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - c3 == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * c3 == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 9>{})>::value / c3 == 3);

  constant_coefficient c3u {M21::Constant(3)};
  EXPECT_EQ(+c3u, 3);
  EXPECT_EQ(-c3u, -3);
  EXPECT_EQ((c3u + std::integral_constant<int, 2>{}), 5);
  EXPECT_EQ((c3u - std::integral_constant<int, 2>{}), 1);
  EXPECT_EQ((c3u * std::integral_constant<int, 2>{}), 6);
  EXPECT_EQ((c3u / std::integral_constant<int, 2>{}), 1.5);

  EXPECT_EQ((std::integral_constant<int, 2>{} + c3u), 5);
  EXPECT_EQ((std::integral_constant<int, 2>{} - c3u), -1);
  EXPECT_EQ((std::integral_constant<int, 2>{} * c3u), 6);
  EXPECT_EQ((std::integral_constant<int, 6>{} / c3u), 2);

  static_assert(constant_diagonal_matrix<Z11, CompileTimeStatus::known>);
  static_assert(constant_diagonal_matrix<Z1x, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z2x, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Zx2, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Zx1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Zxx, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Zx1>);
  static_assert(not constant_diagonal_matrix<Zxx>);
  static_assert(not constant_diagonal_matrix<Z21, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Z12, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Z23, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<C11_m1> == -1);
  static_assert(constant_diagonal_coefficient_v<C11_2> == 2);
  static_assert(constant_diagonal_coefficient_v<C11_m2> == -2);
  static_assert(constant_diagonal_matrix<C11_1>);
  static_assert(not constant_diagonal_matrix<C21_1, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<C2x_1, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<C1x_1>);
  static_assert(not constant_diagonal_matrix<Cx1_1>);
  static_assert(not constant_diagonal_matrix<Cxx_1>);
  static_assert(constant_diagonal_matrix<C11_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C1x_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Cx1_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Cxx_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<I22> == 1);
  static_assert(constant_diagonal_coefficient_v<I2x> == 1);
  static_assert(constant_diagonal_coefficient_v<Cd22_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd2x_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cdx2_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cdxx_2> == 2);

  constant_diagonal_coefficient<Cd22_3> cd3;
  static_assert(std::decay_t<decltype(+cd3)>::value == 3);
  static_assert(std::decay_t<decltype(-cd3)>::value == -3);
  static_assert(std::decay_t<decltype(cd3 + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(cd3 - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(cd3 * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(cd3 / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + cd3 == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - cd3 == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * cd3 == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 9>{})>::value / cd3 == 3);

  constant_diagonal_coefficient cd3u {Eigen::DiagonalWrapper {M21::Constant(3)}};
  EXPECT_EQ(+cd3u, 3);
  EXPECT_EQ(-cd3u, -3);
  EXPECT_EQ((cd3u + std::integral_constant<int, 2>{}), 5);
  EXPECT_EQ((cd3u - std::integral_constant<int, 2>{}), 1);
  EXPECT_EQ((cd3u * std::integral_constant<int, 2>{}), 6);
  EXPECT_EQ((cd3u / std::integral_constant<int, 2>{}), 1.5);

  EXPECT_EQ((std::integral_constant<int, 2>{} + cd3u), 5);
  EXPECT_EQ((std::integral_constant<int, 2>{} - cd3u), -1);
  EXPECT_EQ((std::integral_constant<int, 2>{} * cd3u), 6);
  EXPECT_EQ((std::integral_constant<int, 6>{} / cd3u), 2);

  auto sc3 = internal::ScalarConstant<Likelihood::definitely, double, 3>{};
  auto sco3 = internal::scalar_constant_operation{std::minus<>{}, internal::ScalarConstant<Likelihood::definitely, double, 7>{}, std::integral_constant<int, 4>{}};
  static_assert(std::decay_t<decltype(c3 + cd3)>::value == 6);
  static_assert(std::decay_t<decltype(sc3 - cd3)>::value == 0);
  static_assert(std::decay_t<decltype(c3 * sco3)>::value == 9);
  static_assert(std::decay_t<decltype(sco3 / sc3)>::value == 1);

  static_assert(zero<Z21>);
  static_assert(zero<Eigen::DiagonalWrapper<Z21>>);
  static_assert(zero<Z23>);
  static_assert(zero<Z2x>);
  static_assert(zero<Zx2>);
  static_assert(zero<B22_false>);
  static_assert(not zero<Cd22_2>);
  static_assert(zero<Z11>);
  static_assert(zero<Zxx>);

  static_assert(not identity_matrix<C21_1, Likelihood::maybe>);
  static_assert(not identity_matrix<C2x_1, Likelihood::maybe>);
  static_assert(identity_matrix<I22>);
  static_assert(identity_matrix<I2x, Likelihood::maybe>);
  static_assert(not identity_matrix<Cd22_2, Likelihood::maybe>);
  static_assert(not identity_matrix<Cd22_3, Likelihood::maybe>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<C1x_1>);
  static_assert(not identity_matrix<Cx1_1>);
  static_assert(not identity_matrix<Cxx_1>);
  static_assert(not identity_matrix<C21_1, Likelihood::maybe>);
  static_assert(not identity_matrix<C2x_1, Likelihood::maybe>);
  static_assert(identity_matrix<C1x_1, Likelihood::maybe>);
  static_assert(identity_matrix<Cx1_1, Likelihood::maybe>);
  static_assert(identity_matrix<Cxx_1, Likelihood::maybe>);

  static_assert(identity_matrix<M00>);
  static_assert(not identity_matrix<M0x>);
  static_assert(identity_matrix<M0x, Likelihood::maybe>);
  static_assert(identity_matrix<Mx0, Likelihood::maybe>);
  static_assert(identity_matrix<Mxx, Likelihood::maybe>);
  static_assert(not identity_matrix<M11, Likelihood::maybe>);
  static_assert(not identity_matrix<M1x, Likelihood::maybe>);
  static_assert(not identity_matrix<Mx1, Likelihood::maybe>);

  static_assert(constant_matrix<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, Dimensions<1>>, CompileTimeStatus::unknown>);
  static_assert(constant_matrix<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, std::size_t>, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, std::size_t>, CompileTimeStatus::unknown>);
  static_assert(constant_matrix<internal::FixedSizeAdapter<const Mxx, std::size_t, Dimensions<1>>, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Mxx, std::size_t, Dimensions<1>>, CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const M2x, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);

}

TEST(eigen3, Eigen_check_test_classes)
{
  static_assert(one_dimensional<Zx1, Likelihood::maybe>);
  static_assert(not one_dimensional<Zx1>);
  static_assert(one_dimensional<C11_1>);
  static_assert(one_dimensional<C1x_1, Likelihood::maybe>);
  static_assert(not one_dimensional<C1x_1>);
  static_assert(one_dimensional<Cxx_1, Likelihood::maybe>);
  static_assert(not one_dimensional<Cxx_1>);
  static_assert(one_dimensional<C11_m1>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<M11>>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M1x>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<M1x>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<Mx1>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mxx>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<Mxx>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M22>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M2x>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mx2>, Likelihood::maybe>);
  static_assert(not one_dimensional<Cd22_2, Likelihood::maybe>);
  static_assert(not one_dimensional<Cd2x_2, Likelihood::maybe>);
  static_assert(one_dimensional<EigenWrapper<Eigen::DiagonalWrapper<Cx1_1>>, Likelihood::maybe>);
  static_assert(one_dimensional<EigenWrapper<Eigen::DiagonalWrapper<C1x_1>>, Likelihood::maybe>);

  static_assert(square_shaped<M22>);
  static_assert(square_shaped<M2x, Likelihood::maybe>);
  static_assert(square_shaped<Mx2, Likelihood::maybe>);
  static_assert(square_shaped<Mxx, Likelihood::maybe>);
  static_assert(square_shaped<M11>);
  static_assert(square_shaped<M1x, Likelihood::maybe>);
  static_assert(square_shaped<Mx1, Likelihood::maybe>);

  static_assert(square_shaped<C11_m1, Likelihood::maybe>);
  static_assert(square_shaped<Z22, Likelihood::maybe>);
  static_assert(square_shaped<Z2x, Likelihood::maybe>);
  static_assert(square_shaped<Zx2, Likelihood::maybe>);
  static_assert(square_shaped<Zxx, Likelihood::maybe>);
  static_assert(square_shaped<C22_2, Likelihood::maybe>);
  static_assert(square_shaped<C2x_2, Likelihood::maybe>);
  static_assert(square_shaped<Cx2_2, Likelihood::maybe>);
  static_assert(square_shaped<Cxx_2, Likelihood::maybe>);
  static_assert(square_shaped<DM2, Likelihood::maybe>);
  static_assert(square_shaped<DMx, Likelihood::maybe>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<M11>>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<M1x>>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<Mxx>>);

  static_assert(square_shaped<C11_m1>);
  static_assert(square_shaped<Z22>);
  static_assert(not square_shaped<Z2x>);
  static_assert(not square_shaped<Zx2>);
  static_assert(not square_shaped<Zxx>);
  static_assert(square_shaped<C22_2>);
  static_assert(not square_shaped<C2x_2>);
  static_assert(not square_shaped<Cx2_2>);
  static_assert(not square_shaped<Cxx_2>);
  static_assert(square_shaped<DM2>);
  static_assert(square_shaped<DMx>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<M11>>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<M1x>>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(square_shaped<Eigen::DiagonalWrapper<Mxx>>);

  static_assert(square_shaped<Tlv22>);
  static_assert(square_shaped<Tlv2x, Likelihood::maybe>);
  static_assert(square_shaped<Tlvx2, Likelihood::maybe>);
  static_assert(square_shaped<Tlvxx, Likelihood::maybe>);
  static_assert(not square_shaped<Tlv2x>);
  static_assert(not square_shaped<Tlvx2>);
  static_assert(not square_shaped<Tlvxx>);

  static_assert(square_shaped<Salv22>);
  static_assert(square_shaped<Salv2x, Likelihood::maybe>);
  static_assert(square_shaped<Salvx2, Likelihood::maybe>);
  static_assert(square_shaped<Salvxx, Likelihood::maybe>);
  static_assert(not square_shaped<Salv2x>);
  static_assert(not square_shaped<Salvx2>);
  static_assert(not square_shaped<Salvxx>);

  static_assert(diagonal_matrix<Z22>);
  static_assert(not diagonal_matrix<Z2x>);
  static_assert(not diagonal_matrix<Zx2>);
  static_assert(not diagonal_matrix<Zxx>);
  static_assert(diagonal_matrix<Z22, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z2x, Likelihood::maybe>);
  static_assert(diagonal_matrix<Zx2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Zxx, Likelihood::maybe>);
  static_assert(diagonal_matrix<C11_2>);
  static_assert(diagonal_matrix<I22>);
  static_assert(diagonal_matrix<I2x, Likelihood::maybe>);
  static_assert(diagonal_matrix<Cd22_2>);
  static_assert(diagonal_matrix<Cd2x_2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Cdx2_2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Cdxx_2, Likelihood::maybe>);
  static_assert(diagonal_matrix<DW21>);
  static_assert(diagonal_matrix<DW2x>);
  static_assert(diagonal_matrix<DWx1>);
  static_assert(diagonal_matrix<DWxx>);
  static_assert(not diagonal_matrix<Salv22>);
  static_assert(not diagonal_matrix<Salv2x>);
  static_assert(not diagonal_matrix<Salvx2>);
  static_assert(not diagonal_matrix<Salvxx>);
  static_assert(not diagonal_matrix<Sauv22>);
  static_assert(not diagonal_matrix<Sauv2x>);
  static_assert(not diagonal_matrix<Sauvx2>);
  static_assert(not diagonal_matrix<Sauvxx>);
  static_assert(diagonal_matrix<Sadv22>);
  static_assert(diagonal_matrix<Sadv2x, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadvx2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadvxx, Likelihood::maybe>);
  static_assert(diagonal_matrix<M11>);
  static_assert(not diagonal_matrix<M1x>);
  static_assert(not diagonal_matrix<Mx1>);
  static_assert(not diagonal_matrix<Mxx>);
  static_assert(diagonal_matrix<M11, Likelihood::maybe>);
  static_assert(diagonal_matrix<M1x, Likelihood::maybe>);
  static_assert(diagonal_matrix<Mx1, Likelihood::maybe>);
  static_assert(diagonal_matrix<Mxx, Likelihood::maybe>);

  static_assert(not diagonal_adapter<M11>);
  static_assert(not diagonal_adapter<M1x>);
  static_assert(not diagonal_adapter<Mx1>);
  static_assert(not diagonal_adapter<Mxx>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M2x>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M2x>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M1x>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M1x>, Likelihood::maybe>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<Mxx>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<Mxx>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M2x>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M2x>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M1x>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M1x>, Likelihood::maybe>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<Mxx>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<Mxx>, Likelihood::maybe>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M22>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M22>, Likelihood::maybe>);

  static_assert(triangular_matrix<Z22, TriangleType::lower>);
  static_assert(not triangular_matrix<Z2x, TriangleType::lower>);
  static_assert(not triangular_matrix<Zx2, TriangleType::lower>);
  static_assert(not triangular_matrix<Zxx, TriangleType::lower>);
  static_assert(triangular_matrix<Z2x, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Zx2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Zxx, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<C11_2, TriangleType::lower>);
  static_assert(not triangular_matrix<C22_2, TriangleType::lower>);
  static_assert(not triangular_matrix<C22_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<I22, TriangleType::lower>);
  static_assert(triangular_matrix<I2x, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd22_2, TriangleType::lower>);
  static_assert(triangular_matrix<Cd2x_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Cdx2_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Cdxx_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<DW21, TriangleType::lower>);
  static_assert(triangular_matrix<DW2x, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<DWx1, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<DWxx, TriangleType::lower, Likelihood::maybe>);

  static_assert(triangular_matrix<Z22, TriangleType::upper>);
  static_assert(not triangular_matrix<Z2x, TriangleType::upper>);
  static_assert(not triangular_matrix<Zx2, TriangleType::upper>);
  static_assert(not triangular_matrix<Zxx, TriangleType::upper>);
  static_assert(triangular_matrix<Z2x, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Zx2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Zxx, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<C11_2, TriangleType::upper>);
  static_assert(not triangular_matrix<C22_2, TriangleType::upper>);
  static_assert(not triangular_matrix<C22_2, TriangleType::upper, Likelihood::maybe>);

  static_assert(triangular_matrix<I22, TriangleType::upper>);
  static_assert(triangular_matrix<I2x, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd22_2, TriangleType::upper>);
  static_assert(triangular_matrix<Cd2x_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Cdx2_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Cdxx_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<DW21, TriangleType::upper>);
  static_assert(triangular_matrix<DW2x, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<DWx1, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<DWxx, TriangleType::upper, Likelihood::maybe>);

  static_assert(triangular_matrix<Tlv22, TriangleType::lower>);
  static_assert(triangular_matrix<Tlv2x, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Tlvx2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Tlvxx, TriangleType::lower, Likelihood::maybe>);
  static_assert(not triangular_matrix<Tlv2x, TriangleType::lower>);
  static_assert(not triangular_matrix<Tlvx2, TriangleType::lower>);
  static_assert(not triangular_matrix<Tlvxx, TriangleType::lower>);
  static_assert(not triangular_matrix<Tuv22, TriangleType::lower>);

  static_assert(triangular_matrix<Tuv22, TriangleType::upper>);
  static_assert(triangular_matrix<Tuv2x, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Tuvx2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Tuvxx, TriangleType::upper, Likelihood::maybe>);
  static_assert(not triangular_matrix<Tuv2x, TriangleType::upper>);
  static_assert(not triangular_matrix<Tuvx2, TriangleType::upper>);
  static_assert(not triangular_matrix<Tuvxx, TriangleType::upper>);
  static_assert(not triangular_matrix<Tlv22, TriangleType::upper>);

  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<Z2x, Likelihood::maybe>);
  static_assert(hermitian_matrix<Zx2, Likelihood::maybe>);
  static_assert(hermitian_matrix<Zxx, Likelihood::maybe>);
  static_assert(not hermitian_adapter<Z22>);
  static_assert(not hermitian_adapter<Z2x>);
  static_assert(not hermitian_adapter<Zx2>);
  static_assert(not hermitian_adapter<Zxx>);
  static_assert(hermitian_matrix<C22_2>);
  static_assert(hermitian_matrix<I22>);
  static_assert(hermitian_matrix<I2x, Likelihood::maybe>);
  static_assert(hermitian_matrix<Cd22_2>);
  static_assert(hermitian_matrix<Cd2x_2, Likelihood::maybe>);
  static_assert(hermitian_matrix<Cdx2_2, Likelihood::maybe>);
  static_assert(hermitian_matrix<Cdxx_2, Likelihood::maybe>);
  static_assert(hermitian_matrix<DW21>);
  static_assert(hermitian_matrix<DW2x>);
  static_assert(hermitian_matrix<DWx1>);
  static_assert(hermitian_matrix<DWxx>);
  static_assert(not hermitian_adapter<C22_2>);
  static_assert(not hermitian_adapter<I22>);
  static_assert(not hermitian_adapter<I2x>);
  static_assert(not hermitian_adapter<Cd22_2>);
  static_assert(not hermitian_adapter<Cd2x_2>);
  static_assert(not hermitian_adapter<Cdx2_2>);
  static_assert(not hermitian_adapter<Cdxx_2>);
  static_assert(not hermitian_adapter<DW21>);
  static_assert(not hermitian_adapter<DW2x>);
  static_assert(not hermitian_adapter<DWx1>);
  static_assert(not hermitian_adapter<DWxx>);

  static_assert(hermitian_adapter<nested_object_of_t<Salv22>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_object_of_t<Salv2x>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_object_of_t<Salvx2>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_object_of_t<Salvxx>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_object_of_t<Salv2x>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_object_of_t<Salvx2>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_object_of_t<Salvxx>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<nested_object_of_t<Sauv22>, HermitianAdapterType::lower>);

  static_assert(hermitian_adapter<nested_object_of_t<Sauv22>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_object_of_t<Sauv2x>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_object_of_t<Sauvx2>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_object_of_t<Sauvxx>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_object_of_t<Sauv2x>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_object_of_t<Sauvx2>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_object_of_t<Sauvxx>, HermitianAdapterType::upper>);
  static_assert(not hermitian_adapter<nested_object_of_t<Salv22>, HermitianAdapterType::upper>);

  static_assert(hermitian_adapter<nested_object_of_t<Sadv22>, HermitianAdapterType::any>);
  static_assert(hermitian_adapter<nested_object_of_t<Sadv22>, HermitianAdapterType::any>);

  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Salv22>> == HermitianAdapterType::lower);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Salv22>, nested_object_of_t<Salv22>> == HermitianAdapterType::lower);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Salv2x>, nested_object_of_t<Salvx2>, nested_object_of_t<Salvxx>> == HermitianAdapterType::lower);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Sauv22>> == HermitianAdapterType::upper);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Sauv22>, nested_object_of_t<Sauv22>> == HermitianAdapterType::upper);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Sauv2x>, nested_object_of_t<Sauvx2>, nested_object_of_t<Sauvxx>> == HermitianAdapterType::upper);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Sauv22>, nested_object_of_t<Salv22>> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<nested_object_of_t<Salv22>, nested_object_of_t<Sauv22>> == HermitianAdapterType::any);

  static_assert(maybe_same_shape_as<>);
  static_assert(maybe_same_shape_as<M32>);
  static_assert(maybe_same_shape_as<M32, Mx2, M3x>);
  static_assert(maybe_same_shape_as<M2x, M23, Mx3>);
  static_assert(maybe_same_shape_as<M2x, Z23>);
  static_assert(maybe_same_shape_as<M2x, Mx3>);

  static_assert(same_shape_as<M32, M32>);
  static_assert(not same_shape_as<M32, Mx2>);
  static_assert(not same_shape_as<Mx2, M32>);
  static_assert(same_shape_as<M22, Salv22, M22, Z22>);
}


TEST(eigen3, EigenWrapper)
{
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<M22>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<I22>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<M22&>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<I22&>>()))>);

  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalMatrix<double, 3>&>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalWrapper<M31>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalWrapper<M31>&>>()))>);

  static_assert(std::is_same_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>&&>())), Eigen::DiagonalMatrix<double, 3>&&>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);

  static_assert(Eigen3::eigen_general<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>);
  static_assert(Eigen3::eigen_wrapper<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>);
}


TEST(eigen3, Eigen_Matrix)
{
  static_assert(Eigen3::eigen_matrix_general<M11, true>);
  static_assert(Eigen3::eigen_matrix_general<Mxx, true>);
  static_assert(Eigen3::eigen_general<M11, true>);
  static_assert(Eigen3::eigen_general<Mxx, true>);

  static_assert(index_count_v<M11> == 0);
  static_assert(index_count_v<Mxx> == 2);
  static_assert(index_count_v<M21> == 1);
  static_assert(index_count_v<Mx1> == 1);
  static_assert(index_count_v<M1x> == 2);
  static_assert(index_count_v<M01> == 1);
  static_assert(index_count_v<M00> == 2);
  static_assert(index_count_v<M20> == 2);
  static_assert(index_count_v<M02> == 2);

  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, Dimensions<2>>> == 2);
  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, Dimensions<1>>> == 1);
  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, Dimensions<2>>> == 2);
  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, Dimensions<1>>> == 0);

  // The orders of these empty matrices are considered to be 0 for now:
  static_assert(max_tensor_order_v<M00> == 0);
  static_assert(max_tensor_order_v<M02> == 0);
  static_assert(max_tensor_order_v<M20> == 0);
  static_assert(max_tensor_order_v<M01> == 0);
  static_assert(max_tensor_order_v<M10> == 0);

  static_assert(max_tensor_order_v<M23> == 2);
  static_assert(max_tensor_order_v<M21> == 1);
  static_assert(max_tensor_order_v<M11> == 0);
  static_assert(max_tensor_order_v<M2x> == 2);
  static_assert(max_tensor_order_v<Mx2> == 2);
  static_assert(max_tensor_order_v<M1x> == 1);
  static_assert(max_tensor_order_v<Mx1> == 1);
  static_assert(max_tensor_order_v<Mxx> == 2);

  static_assert(index_dimension_of_v<M11, 0> == 1);
  static_assert(index_dimension_of_v<M21, 0> == 2);
  static_assert(index_dimension_of_v<Mxx, 0> == dynamic_size);
  static_assert(index_dimension_of_v<M11, 1> == 1);
  static_assert(index_dimension_of_v<M21, 1> == 1);
  static_assert(index_dimension_of_v<Mxx, 1> == dynamic_size);
  EXPECT_EQ(get_vector_space_descriptor<0>(M11{}), 1);
  EXPECT_EQ(get_vector_space_descriptor<0>(M21{}), 2);
  EXPECT_EQ((get_vector_space_descriptor<0>(Mxx{2, 1})), 2);
  EXPECT_EQ((get_vector_space_descriptor<1>(M11{})), 1);
  EXPECT_EQ((get_vector_space_descriptor<1>(M21{})), 1);
  EXPECT_EQ((get_vector_space_descriptor<1>(Mxx{2, 1})), 1);

  static_assert(std::is_same_v<typename interface::indexible_object_traits<Mxx>::scalar_type, double>);

  static_assert(dynamic_dimension<eigen_matrix_t<double, dynamic_size, dynamic_size>, 0>);
  static_assert(dynamic_dimension<eigen_matrix_t<double, dynamic_size, dynamic_size>, 1>);
  static_assert(dynamic_dimension<eigen_matrix_t<double, dynamic_size, 1>, 0>);
  static_assert(not dynamic_dimension<eigen_matrix_t<double, dynamic_size, 1>, 1>);
  static_assert(not dynamic_dimension<eigen_matrix_t<double, 1, dynamic_size>, 0>);
  static_assert(dynamic_dimension<eigen_matrix_t<double, 1, dynamic_size>, 1>);

  static_assert(dynamic_index_count_v<M22> == 0);
  static_assert(dynamic_index_count_v<M2x> == 1);
  static_assert(dynamic_index_count_v<Mx2> == 1);
  static_assert(dynamic_index_count_v<Mxx> == 2);

  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, Dimensions<2>>> == 0);
  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, std::size_t>> == 1);
  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx, std::size_t, std::size_t>> == 2);
  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx>> == 0);

  static_assert(std::is_same_v<dense_writable_matrix_t<M33>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<M3x>, M3x>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Mx3>, Mx3>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Mxx>, Mxx>);

  static_assert(writable<M22>);
  static_assert(writable<M2x>);
  static_assert(writable<Mx2>);
  static_assert(writable<Mxx>);
  static_assert(writable<M22&>);
  static_assert(not writable<const M22>);
  static_assert(not writable<const M22&>);
  static_assert(writable<dense_writable_matrix_t<M22>>);

  static_assert(interface::get_component_defined_for<M32, M32, std::array<std::size_t, 2>>);

  static_assert(element_gettable<M32, 2>);
  static_assert(element_gettable<const M32, 2>);
  static_assert(element_gettable<M31, 2>);
  static_assert(element_gettable<M13, 2>);
  static_assert(element_gettable<M3x, 2>);
  static_assert(element_gettable<Mx2, 2>);
  static_assert(element_gettable<Mx1, 2>);
  static_assert(element_gettable<M1x, 2>);
  static_assert(element_gettable<Mxx, 2>);

  static_assert(element_gettable<M32, 2>);
  static_assert(element_gettable<M31, 1>);
  static_assert(element_gettable<M13, 2>);
  static_assert(element_gettable<Mx2, 2>);
  static_assert(element_gettable<Mx1, 1>);
  static_assert(element_gettable<M1x, 2>);
  static_assert(element_gettable<Mxx, 2>);

  static_assert(element_settable<M32, 2>);
  static_assert(element_settable<M32&&, 2>);
  static_assert(not element_settable<const M32&, 2>);
  static_assert(element_settable<M31&, 2>);
  static_assert(element_settable<M13&, 2>);
  static_assert(element_settable<M3x&, 2>);
  static_assert(element_settable<Mx2&, 2>);
  static_assert(element_settable<Mx1&, 2>);
  static_assert(element_settable<M1x&, 2>);
  static_assert(element_settable<Mxx&, 2>);

  static_assert(element_settable<M32&, 2>);
  static_assert(element_settable<M31&, 1>);
  static_assert(element_settable<M13&, 2>);
  static_assert(not element_settable<const M31&, 1>);
  static_assert(element_settable<Mx2&, 2>);
  static_assert(element_settable<Mx1&, 1>);
  static_assert(element_settable<M1x&, 2>);
  static_assert(element_settable<Mxx&, 2>);

  M22 m22; m22 << 1, 2, 3, 4;
  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  Mx3 mx3_2 {2,3}; mx3_2 << 1, 2, 3, 4, 5, 6;
  M32 m32; m32 << 1, 2, 3, 4, 5, 6;
  CM22 cm22; cm22 << cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1};

  EXPECT_TRUE(is_near(make_dense_object_from<M22>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<M2x>(1, 2)), 1> == 1);
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<Mx2>(1, 2)), 0> == 1);

  static_assert(std::is_same_v<vector_space_descriptor_of<M11, 0>::type, Dimensions<1>>);
  static_assert(std::is_same_v<vector_space_descriptor_of<M11, 1>::type, Dimensions<1>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M11, 0>, Axis>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M11, 1>, Axis>);
  static_assert(std::is_same_v<vector_space_descriptor_of<M22, 0>::type, Dimensions<2>>);
  static_assert(std::is_same_v<vector_space_descriptor_of<M22, 1>::type, Dimensions<2>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M22, 0>, TypedIndex<Axis, Axis>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M22, 1>, TypedIndex<Axis, Axis>>);

  static_assert(maybe_same_shape_as<M22, M2x, Mx2, Mxx>);
  static_assert(same_shape_as<M22, CM22, M22>);
  EXPECT_TRUE(same_shape(m22, cm22, M2x{m22}, Mx2{m22}, Mxx{m22}));

  static_assert(compatible_with_vector_space_descriptors<M23, std::integral_constant<int, 2>, std::integral_constant<int, 3>>);
  static_assert(not compatible_with_vector_space_descriptors<M23, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<M23, int, int>);
  static_assert(compatible_with_vector_space_descriptors<M2x, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<M2x, std::integral_constant<int, 2>, int>);
  static_assert(compatible_with_vector_space_descriptors<M2x, int, int>);
  static_assert(not compatible_with_vector_space_descriptors<M2x, std::integral_constant<int, 3>, int>);
  static_assert(compatible_with_vector_space_descriptors<Mx2, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<Mx2, std::integral_constant<int, 2>, int>);
  static_assert(compatible_with_vector_space_descriptors<Mx2, int, int>);
  static_assert(not compatible_with_vector_space_descriptors<Mx2, int, std::integral_constant<int, 3>>);
  static_assert(compatible_with_vector_space_descriptors<Mxx, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<Mxx, int, int>);

  static_assert(empty_object<M00>);
  static_assert(empty_object<M01>);
  static_assert(empty_object<M10>);
  static_assert(empty_object<M0x>);
  static_assert(empty_object<Mx0>);
  static_assert(not empty_object<M11>);
  static_assert(not empty_object<M1x>);
  static_assert(not empty_object<Mx1>);
  static_assert(not empty_object<Mxx>);
  static_assert(empty_object<M1x, Likelihood::maybe>);
  static_assert(empty_object<Mx1, Likelihood::maybe>);
  static_assert(empty_object<Mxx, Likelihood::maybe>);

  static_assert(one_dimensional<M11>);
  static_assert(not one_dimensional<M1x>);
  static_assert(one_dimensional<M1x, Likelihood::maybe>);
  static_assert(not one_dimensional<M00>);
  static_assert(not one_dimensional<M01>);
  static_assert(not one_dimensional<M10>);
  static_assert(not one_dimensional<M00, Likelihood::maybe>);
  static_assert(not one_dimensional<Mx0, Likelihood::maybe>);
  static_assert(not one_dimensional<M0x, Likelihood::maybe>);

  static_assert(square_shaped<M00, Likelihood::maybe>);
  static_assert(square_shaped<M0x, Likelihood::maybe>);
  static_assert(square_shaped<Mx0, Likelihood::maybe>);
  static_assert(square_shaped<M11, Likelihood::maybe>);
  static_assert(square_shaped<M22, Likelihood::maybe>);
  static_assert(not square_shaped<M32, Likelihood::maybe>);
  static_assert(square_shaped<M2x, Likelihood::maybe>);
  static_assert(square_shaped<Mx2, Likelihood::maybe>);
  static_assert(square_shaped<Mxx, Likelihood::maybe>);
  static_assert(square_shaped<CM22, Likelihood::maybe>);
  static_assert(not square_shaped<CM32, Likelihood::maybe>);
  static_assert(square_shaped<CM2x, Likelihood::maybe>);
  static_assert(square_shaped<CMx2, Likelihood::maybe>);
  static_assert(square_shaped<CMxx, Likelihood::maybe>);

  static_assert(square_shaped<M00>);
  static_assert(not square_shaped<M0x>);
  static_assert(not square_shaped<Mx0>);
  static_assert(square_shaped<M11>);
  static_assert(square_shaped<M22>);
  static_assert(not square_shaped<M2x>);
  static_assert(not square_shaped<Mx2>);
  static_assert(not square_shaped<Mxx>);
  static_assert(square_shaped<CM22>);
  static_assert(not square_shaped<CM2x>);
  static_assert(not square_shaped<CMx2>);
  static_assert(not square_shaped<CMxx>);

  M11 m11_1{1};
  Eigen::Matrix<double, 0, 0> m00;

  static_assert(is_square_shaped(m00));
  static_assert(is_square_shaped(m22));
  static_assert(*is_square_shaped(m22) == Dimensions<2>{});
  EXPECT_TRUE(is_square_shaped(M2x{m22}));
  EXPECT_TRUE(*is_square_shaped(M2x{m22}) == Dimensions<2>{});
  EXPECT_TRUE(is_square_shaped(Mx2{m22}));
  EXPECT_TRUE(*is_square_shaped(Mx2{m22}) == Dimensions<2>{});
  EXPECT_TRUE(is_square_shaped(Mxx{m22}));
  EXPECT_EQ(*is_square_shaped(Mxx{m22}), 2);
  static_assert(is_square_shaped(m11_1));
  static_assert(*is_square_shaped(m11_1) == Dimensions<1>{});
  EXPECT_TRUE(is_square_shaped(M1x{m11_1}));
  EXPECT_TRUE(*is_square_shaped(M1x{m11_1}) == Dimensions<1>{});
  EXPECT_TRUE(is_square_shaped(Mx1{m11_1}));
  EXPECT_TRUE(*is_square_shaped(Mx1{m11_1}) == Dimensions<1>{});
  EXPECT_TRUE(is_square_shaped(Mxx{m11_1}));
  EXPECT_EQ(*is_square_shaped(Mxx{m11_1}), 1);

  static_assert(is_one_dimensional(m11_1));
  EXPECT_TRUE(is_one_dimensional(M1x{m11_1}));
  EXPECT_TRUE(is_one_dimensional(Mx1{m11_1}));
  EXPECT_TRUE(is_one_dimensional(Mxx{m11_1}));
  static_assert(not is_one_dimensional(m00));

  static_assert(dimension_size_of_index_is<M31, 1, 1>);
  static_assert(dimension_size_of_index_is<Mx1, 1, 1>);
  static_assert(dimension_size_of_index_is<M3x, 1, 1, Likelihood::maybe>);
  static_assert(dimension_size_of_index_is<Mxx, 1, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M32, 1, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<Mx2, 1, 1, Likelihood::maybe>);

  static_assert(dimension_size_of_index_is<M13, 0, 1>);
  static_assert(dimension_size_of_index_is<M1x, 0, 1>);
  static_assert(dimension_size_of_index_is<Mx3, 0, 1, Likelihood::maybe>);
  static_assert(dimension_size_of_index_is<Mxx, 0, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M23, 0, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M2x, 0, 1, Likelihood::maybe>);

  M21 m21 {1, 2};
  M2x m2x_1 {m21};
  Mx1 mx1_2 {m21};
  Mxx mxx_21 {m21};
  M12 m12 {1, 2};
  M1x m1x_2 {m12};
  Mx2 mx2_1 {m12};
  Mxx mxx_12 {m12};

  static_assert(vector<M11>);
  static_assert(is_vector(m11_1));

  static_assert(vector<M21>);
  static_assert(vector<M2x, 0, Likelihood::maybe>);
  static_assert(vector<Mx1, 0, Likelihood::maybe>);
  static_assert(vector<Mxx, 0, Likelihood::maybe>);
  static_assert(not vector<M12, 0, Likelihood::maybe>);
  static_assert(not vector<Mx2, 0, Likelihood::maybe>);
  static_assert(is_vector(m21));
  EXPECT_TRUE(is_vector(m2x_1));
  static_assert(is_vector(mx1_2));
  EXPECT_TRUE(is_vector(mxx_21));
  EXPECT_FALSE(is_vector(mxx_12));

  static_assert(vector<M12, 1>);
  static_assert(vector<M1x, 1, Likelihood::maybe>);
  static_assert(vector<Mx2, 1, Likelihood::maybe>);
  static_assert(vector<Mxx, 1, Likelihood::maybe>);
  static_assert(not vector<M21, 1, Likelihood::maybe>);
  static_assert(not vector<M2x, 1, Likelihood::maybe>);
  static_assert(is_vector<1>(m12));
  static_assert(is_vector<1>(m1x_2));
  EXPECT_TRUE(is_vector<1>(mx2_1));
  EXPECT_TRUE(is_vector<1>(mxx_12));
  EXPECT_FALSE(is_vector<1>(mxx_21));

  static_assert(eigen_matrix_general<M22, true>);
  static_assert(eigen_matrix_general<Mxx, true>);
  static_assert(eigen_matrix_general<CM22, true>);
  static_assert(eigen_matrix_general<CMxx, true>);
  static_assert(not eigen_matrix_general<double, true>);

  //static_assert(modifiable<M33, M33>);
  //static_assert(not modifiable<M33, M31>);
  //static_assert(not modifiable<M33, eigen_matrix_t<int, 3, 3>>);
  //static_assert(not modifiable<const M33, M33>);
  //static_assert(modifiable<M33, Eigen3::IdentityMatrix<M33>>);
}


TEST(eigen3, Eigen_Array)
{
  static_assert(eigen_array_general<Eigen::Array<double, 3, 2, true>>);
  static_assert(not eigen_matrix_general<Eigen::Array<double, 3, 2, true>>);
  static_assert(self_contained<Eigen::Array<double, 3, 2>>);
  static_assert(index_dimension_of_v<Eigen::Array<double, 3, 2>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::Array<double, 3, 2>, 1> == 2);
  static_assert(not square_shaped<Eigen::Array<double, 2, 1>>);
}


TEST(eigen3, Eigen_ArrayWrapper)
{
  static_assert(eigen_array_general<Eigen::ArrayWrapper<M32>, true>);
  static_assert(not eigen_matrix_general<Eigen::ArrayWrapper<M32>, true>);
  static_assert(self_contained<Eigen::ArrayWrapper<I22>>);
  static_assert(not self_contained<Eigen::ArrayWrapper<M32>>);

  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::ArrayWrapper<M32>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::ArrayWrapper<I22>>()))>);

  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C22_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C2x_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<Cx2_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<Cxx_2>> == 2);

  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd22_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd2x_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cdx2_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cdxx_2>> == 2);

  static_assert(zero<Eigen::ArrayWrapper<Z22>>);
  static_assert(zero<Eigen::ArrayWrapper<Z21>>);
  static_assert(zero<Eigen::ArrayWrapper<Z23>>);

  static_assert(diagonal_matrix<DW21>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<Z22>>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<C22_2>>);
  static_assert(triangular_matrix<DW21, TriangleType::lower>);
  static_assert(triangular_matrix<DW21, TriangleType::upper>);
}


TEST(eigen3, Eigen_Block)
{
  static_assert(eigen_matrix_general<decltype(std::declval<C22_2>().matrix().block<2,1>(0, 0)), true>);
  static_assert(eigen_array_general<decltype(std::declval<C22_2>().block<2,1>(0, 0)), true>);
  static_assert(not eigen_matrix_general<decltype(std::declval<C22_2>().block<2,1>(0, 0)), true>);

  static_assert(self_contained<decltype(std::declval<I22>().block<2,1>(0, 0))>);
  static_assert(not self_contained<decltype(std::declval<Eigen::ArrayWrapper<M32>>().block<2,1>(0, 0))>);
  static_assert(self_contained<decltype((2 * std::declval<I22>() + std::declval<I22>()).col(0))>);
  static_assert(not self_contained<decltype((2 * std::declval<I22>() + A22 {1, 2, 3, 4}).col(0))>);
  static_assert(self_contained<decltype((2 * std::declval<I22>() + std::declval<I22>()).row(0))>);
  static_assert(not self_contained<decltype((2 * std::declval<I22>() + A22 {1, 2, 3, 4}).row(0))>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().block<2, 1>(0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().block(2, 1, 0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 2>(0, 0))> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().block<1, 1>(0, 0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().block<2, 2>(0, 0))> == 0);

  static_assert(zero<decltype(std::declval<Z22>().block<2, 1>(0, 0))>);
  static_assert(zero<decltype(std::declval<Z22>().block(2, 1, 0, 0))>);
  static_assert(zero<decltype(std::declval<Z22>().block<1, 1>(0, 0))>);

  static_assert(writable<Eigen::Block<M33, 3, 1, true>>);
  static_assert(writable<Eigen::Block<M33, 3, 2, true>>);

  static_assert(modifiable<Eigen::Block<M33, 3, 1, true>, M31>);
  static_assert(modifiable<Eigen::Block<M33, 3, 1, true>, Eigen::Block<M33, 3, 1, true>>);
  static_assert(not modifiable<Eigen::Block<M33, 3, 2, true>, M31>);
  static_assert(not modifiable<Eigen::Block<M33, 3, 2, true>, Eigen::Block<M33, 3, 1, true>>);
}

// For Eigen::CwiseBinaryOp, see cwise_binary_operations

// For Eigen::CwiseNullaryOp, see cwise_nullary_operations

// For Eigen::CwiseTernaryOp, see cwise_ternary_operations

// For Eigen::CwiseUnaryOp, see cwise_unary_operations

// For Eigen::CwiseUnaryView, see cwise_unary_operations


TEST(eigen3, Eigen_Diagonal)
{
  static_assert(not self_contained<decltype(std::declval<M22>().diagonal())>);
  static_assert(self_contained<decltype(std::declval<C22_2>().matrix().diagonal())>);

  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(M2x::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(Mx2::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(Mxx::Identity().diagonal())> == 1);

  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(M2x::Identity().diagonal<-1>())> == 0);
  static_assert(constant_coefficient_v<decltype(Mx2::Identity().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(Mxx::Identity().diagonal<-1>())> == 0);
  static_assert(constant_matrix<decltype(M22::Identity().diagonal<Eigen::DynamicIndex>()), CompileTimeStatus::unknown>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<-1>())> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C1x_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cx1_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().matrix().diagonal())> == 2);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_2>().matrix().diagonal<1>())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().matrix().diagonal())>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd2x_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd2x_2>().matrix().diagonal<1>())> == 0);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd2x_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cdx2_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cdxx_2>().matrix().diagonal())>);

  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, -2>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, -1>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 0>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 1>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 2>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 3>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, Eigen::DynamicIndex>, 0, dynamic_size>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, 2>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, 1>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, 0>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, -1>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, -2>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, -3>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, Eigen::DynamicIndex>, 0, dynamic_size>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M2x, 0>, 0, dynamic_size>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<Mx2, 0>, 0, dynamic_size>);

  static_assert(one_dimensional<Eigen::Diagonal<M11, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M13, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M13, 1>>);
  static_assert(not one_dimensional<Eigen::Diagonal<M1x, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M1x, 0>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Diagonal<M31, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M31, -1>>);
  static_assert(not one_dimensional<Eigen::Diagonal<Mx1, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<Mx1, 0>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Diagonal<M22, Eigen::DynamicIndex>>);
  static_assert(one_dimensional<Eigen::Diagonal<M22, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Diagonal<M2x, 0>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Diagonal<Mx2, 0>, Likelihood::maybe>);

  static_assert(dimension_size_of_index_is<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, Dimensions<2>, Dimensions<1>>, 0, 2>);
  static_assert(dimension_size_of_index_is<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, Dimensions<2>, Dimensions<1>>, 1, 1>);

  static_assert(not one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, Dimensions<2>, Dimensions<1>>, Likelihood::maybe>);
  static_assert(not one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, Dimensions<2>, Dimensions<1>>, Likelihood::maybe>);
  static_assert(not one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<Mx2, 0>, Dimensions<2>, Dimensions<1>>, Likelihood::maybe>);
  static_assert(one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, Dimensions<1>, Dimensions<1>>>);

  static_assert(not square_shaped<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, Dimensions<2>, Dimensions<1>>, Likelihood::maybe>);

  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mx2, 0>, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, Dimensions<1>, Dimensions<1>>, CompileTimeStatus::unknown>);

  static_assert(not constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mx2, 0>, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, Dimensions<2>, Dimensions<1>>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, Dimensions<1>, Dimensions<1>>, CompileTimeStatus::unknown>);
}


TEST(eigen3, Eigen_DiagonalMatrix)
{
  static_assert(index_dimension_of_v<DM2, 0> == 2);
  static_assert(index_dimension_of_v<DMx, 0> == dynamic_size);

  static_assert(index_dimension_of_v<DM2, 1> == 2);
  static_assert(index_dimension_of_v<DMx, 1> == dynamic_size);

  static_assert(self_contained<DM2>);
  static_assert(self_contained<DMx>);

  static_assert(square_shaped<DMx>);

  static_assert(diagonal_matrix<DM2>);
  static_assert(diagonal_matrix<DMx>);

  static_assert(triangular_matrix<DMx, TriangleType::lower>);

  static_assert(std::is_same_v<nested_object_of_t<Eigen::DiagonalMatrix<double, 2>>, M21&&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, 3>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>, Mxx>);

  static_assert(diagonal_matrix<Eigen::DiagonalMatrix<double, 3>>);

  static_assert(not writable<Eigen::DiagonalMatrix<double, 3>>);

  /*static_assert(element_gettable<Eigen::DiagonalMatrix<double, 2>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalMatrix<double, 2>, 1>);
  static_assert(element_gettable<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 1>);

  static_assert(element_gettable<Eigen::DiagonalWrapper<M21>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalWrapper<M21>, 1>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M2x>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalWrapper<M2x>, 1>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<Mx1>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalWrapper<Mx1>, 1>);*/

  static_assert(Eigen3::eigen_general<Eigen::DiagonalMatrix<double, 3>, true>);
  static_assert(not Eigen3::eigen_dense_general<Eigen::DiagonalMatrix<double, 3>, true>);
}


TEST(eigen3, Eigen_DiagonalWrapper)
{
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M31>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M3x>, 0> == dynamic_size);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx1>, 0> == dynamic_size);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mxx>, 0> == dynamic_size);

  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M13>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M1x>, 0> == dynamic_size);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx3>, 0> == dynamic_size);

  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M31>, 1> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M3x>, 1> == dynamic_size);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx1>, 1> == dynamic_size);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mxx>, 1> == dynamic_size);

  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M13>, 1> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M1x>, 1> == dynamic_size);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx3>, 1> == dynamic_size);

  static_assert(square_shaped<Eigen::DiagonalWrapper<Mxx>>);

  static_assert(std::is_same_v<nested_object_of_t<Eigen::DiagonalWrapper<M21>>, M21&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M31>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M22>>, M44>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M3x>>, Mxx>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<Mx1>>, Mxx>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<Mxx>>, Mxx>);

  static_assert(not self_contained<Eigen::DiagonalWrapper<M31>>);

  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix().asDiagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>().matrix().asDiagonal())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C21_2>().matrix().asDiagonal())>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix().asDiagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z21>().matrix().asDiagonal())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().matrix().asDiagonal())> == 2);

  static_assert(not zero<decltype(std::declval<C11_1>())>);
  static_assert(not zero<decltype(std::declval<C11_2>())>);
  static_assert(not zero<decltype(std::declval<C21_2>())>);

  static_assert(zero<decltype(std::declval<Z11>().matrix().asDiagonal())>);
  static_assert(zero<decltype(std::declval<Z21>().matrix().asDiagonal())>);

  static_assert(identity_matrix<decltype(std::declval<C11_1>().matrix().asDiagonal())>);
  static_assert(identity_matrix<decltype(std::declval<C21_1>().matrix().asDiagonal())>);

  static_assert(diagonal_matrix<Eigen::DiagonalWrapper<M31>>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_2>().matrix().asDiagonal())>);
  static_assert(diagonal_matrix<decltype(std::declval<C21_2>().matrix().asDiagonal())>);

  static_assert(not writable<Eigen::DiagonalWrapper<M31>>);
}


// No current tests for Eigen::Homogeneous


#if EIGEN_VERSION_AT_LEAST(3,4,0)
// No current tests for Eigen::IndexedView
#endif


// No current tests for Eigen::Inverse


// No current tests for Eigen::Map


TEST(eigen3, Eigen_MatrixWrapper)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix())> == 2);
  static_assert(zero<decltype(std::declval<Z23>().matrix())>);
  static_assert(identity_matrix<decltype(std::declval<I22>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().matrix())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().matrix()), TriangleType::upper>);

  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::MatrixWrapper<M32>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::MatrixWrapper<I22>>()))>);
}


// No current tests for Eigen::PermutationWrapper.


TEST(eigen3, Eigen_Product)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_m2>().matrix())> == -8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().matrix() * std::declval<C22_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Cx2_2>().matrix())> == 8);
  static_assert(constant_matrix<decltype(std::declval<C2x_2>().matrix() * std::declval<Cx2_2>().matrix()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<I22>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<I22>().matrix() * std::declval<C22_2>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);

  static_assert(scalar_constant<std::integral_constant<long long unsigned int, 2>>);
  static_assert(not OpenKalman::detail::internal_constant<std::integral_constant<long long unsigned int, 2>>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix() * std::declval<Cd22_3>().matrix())> == 6);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix() * std::declval<I22>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<I22>().matrix() * std::declval<Cd22_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);

  static_assert(zero<decltype(std::declval<Z11>().matrix() * std::declval<Z11>().matrix())>);
  static_assert(zero<decltype(std::declval<Z12>().matrix() * std::declval<Z21>().matrix())>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().matrix() * std::declval<DW21>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Salv22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Sauv22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(triangular_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Tlv22>().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().matrix() * std::declval<Cd22_2>().matrix()), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Tuv22>().matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().matrix() * std::declval<Cd22_2>().matrix()), TriangleType::upper>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().matrix() * std::declval<Tlv22>().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().matrix() * std::declval<Tuv22>().matrix()), TriangleType::upper>);
}


TEST(eigen3, Eigen_Ref)
{
  static_assert(constant_coefficient_v<Eigen::Ref<C22_m2>> == -2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Ref<Cd22_2>> == 2);
}


TEST(eigen3, Eigen_Replicate)
{
  auto z11 = M11::Identity() - M11::Identity();
  using Z11 = decltype(z11);
  auto z20_1 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 1};
  auto z01_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 1> {z11, 2, 1};
  auto z00_21 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 1};
  auto z22 = Eigen::Replicate<Z11, 2, 2> {z11};
  auto z20_2 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  static_assert(Eigen3::eigen_general<Zxx, true>);
  static_assert(index_count_v<Zxx> == 2);
  static_assert(index_dimension_of_v<Zxx, 0> == dynamic_size);
  static_assert(index_dimension_of_v<Zxx, 1> == dynamic_size);
  EXPECT_EQ(get_vector_space_descriptor<0>(z00_21), 2);
  EXPECT_EQ(get_vector_space_descriptor<1>(z00_21), 1);
  static_assert(std::is_same_v<typename interface::indexible_object_traits<Zxx>::scalar_type, double>);

  static_assert(one_dimensional<Eigen::Replicate<M11, 1, 1>>);
  static_assert(one_dimensional<Eigen::Replicate<Mxx, 1, 1>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Replicate<M1x, 1, 1>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Replicate<Mx1, 1, 1>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Replicate<M2x, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Replicate<Mx2, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Replicate<Mxx, 1, 1>>);
  static_assert(one_dimensional<Eigen::Replicate<M11, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Replicate<M11, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(square_shaped<Eigen::Replicate<M22, 3, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M22, Eigen::Dynamic, 3>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<M22, Eigen::Dynamic, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M22, 3, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<M22, 3, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Replicate<M32, 2, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M32, Eigen::Dynamic, 3>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<M32, Eigen::Dynamic, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M32, 2, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<M32, 2, Eigen::Dynamic>>);
  static_assert(not square_shaped<Eigen::Replicate<M32, 5, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<M32, Eigen::Dynamic, 2>, Likelihood::maybe>);
  static_assert(square_shaped<Eigen::Replicate<M3x, 2, 3>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<M3x, 2, 3>>);
  static_assert(square_shaped<Eigen::Replicate<Mx2, 2, 3>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<Mx2, 2, 3>>);
  static_assert(not square_shaped<Eigen::Replicate<M2x, 2, 3>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Replicate<Mx3, 2, 3>, Likelihood::maybe>);

  static_assert(constant_coefficient_v<Eigen::Replicate<Z11, 1, 2>> == 0);
  static_assert(constant_coefficient_v<decltype(z20_1)> == 0);
  static_assert(constant_coefficient_v<decltype(z01_2)> == 0);
  static_assert(constant_coefficient_v<Eigen::Replicate<C2x_2, 1, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Replicate<Cx2_2, 1, 2>> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().replicate<5,5>())> == 2);
  static_assert(not constant_matrix<Eigen::Replicate<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::any, Likelihood::maybe>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().replicate<1,1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().replicate<5,5>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(z22)> == 0);
  static_assert(not constant_diagonal_matrix<decltype(z20_2)>);
  static_assert(not constant_diagonal_matrix<decltype(z02_2)>);
  static_assert(not constant_diagonal_matrix<decltype(z00_22)>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().replicate<1,1>())> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Replicate<C22_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::any, Likelihood::maybe>);

  static_assert(identity_matrix<Eigen::Replicate<I22, 1, 1>>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().replicate<1, 1>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<2, 2>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<2, Eigen::Dynamic>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, 2>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, Eigen::Dynamic>())>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().replicate<1, 1>()), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().replicate<1, 1>()), TriangleType::upper>);

  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().replicate<1, 1>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().replicate<1, 1>())>);
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_Reshaped)
{
  static_assert(eigen_matrix_general<Eigen::Reshaped<M32, 2, 3, Eigen::RowMajor>, true>);
  static_assert(eigen_array_general<Eigen::Reshaped<Eigen::ArrayWrapper<M32>, 2, 3, Eigen::RowMajor>, true>);
  static_assert(self_contained<Eigen::Reshaped<I22, 1, 2, Eigen::RowMajor>>);
  static_assert(self_contained<Eigen::Reshaped<I22, 1, 2, Eigen::ColMajor>>);
  static_assert(self_contained<Eigen::Reshaped<I22, Eigen::Dynamic, 2, Eigen::RowMajor>>);
  static_assert(self_contained<Eigen::Reshaped<Ixx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);
  static_assert(not self_contained<Eigen::Reshaped<M32, 2, 3, Eigen::RowMajor>>);
  static_assert(not self_contained<Eigen::Reshaped<M32, 2, Eigen::Dynamic, Eigen::RowMajor>>);
  static_assert(not self_contained<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);
  auto m32 = make_eigen_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6);
  auto m23 = make_eigen_matrix<double, 2, 3>(1, 4, 2, 5, 3, 6);
  EXPECT_TRUE(is_near(make_self_contained(m32.reshaped<Eigen::RowMajor>(2, 3)), m23));

  static_assert(index_dimension_of_v<Eigen::Reshaped<Mxx, 3, Eigen::Dynamic>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Mxx, Eigen::Dynamic, 4>, 1> == 4);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M21, Eigen::Dynamic, 2>, 0> == 1);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M21, 1, Eigen::Dynamic>, 1> == 2);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M12, Eigen::Dynamic, 1>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M12, 2, Eigen::Dynamic>, 1> == 1);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, 4>, 0> == 4);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, 4, Eigen::Dynamic>, 1> == 4);

  static_assert(one_dimensional<Eigen::Reshaped<Mxx, 1, 1>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Reshaped<M11, 1, 1>>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mxx, 2, 2>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<M2x, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx2, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<M2x, Eigen::Dynamic, 1>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx2, 1, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<M2x, 1, 1>, Likelihood::maybe>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx2, 1, 1>, Likelihood::maybe>);

  static_assert(square_shaped<Eigen::Reshaped<Mxx, 2, 2>>);
  static_assert(square_shaped<Eigen::Reshaped<Mxx, 2, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mxx, 2, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, 2>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, 2>>);
  static_assert(square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<M11, 1, 1>>);
  static_assert(square_shaped<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Mx2, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, 4, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, 4>>);
  static_assert(square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(not square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 9>, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(not square_shaped<Eigen::Reshaped<M5x, 2, 2>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx5, 2, 2>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, 1, 1>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, 1, 1>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, 1>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, 1, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<M21, Eigen::Dynamic, 1>, Likelihood::maybe>);
  static_assert(not square_shaped<Eigen::Reshaped<M21, 2, Eigen::Dynamic>, Likelihood::maybe>);

  static_assert(constant_coefficient_v<Eigen::Reshaped<C22_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C2x_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<Cx2_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<Cxx_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C22_2, 4, 1>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C2x_2, 1, 4>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<Cx2_2, Eigen::Dynamic, 1>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<Cxx_2, 1, Eigen::Dynamic>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<Cxx_2, Eigen::Dynamic, Eigen::Dynamic>> == 2);

  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd22_2, 2, 2>> == 2);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, 2, 2>, CompileTimeStatus::known>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd2x_2, 2, Eigen::Dynamic>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd2x_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, 2>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, 2>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cdx2_2, 2, Eigen::Dynamic>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdx2_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cdxx_2, 2, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, 2, Eigen::Dynamic>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, 2>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, 2>>);

  static_assert(zero<Eigen::Reshaped<Z22, 4, 1>>);
  static_assert(zero<Eigen::Reshaped<Z21, 1, 2>>);
  static_assert(zero<Eigen::Reshaped<Z23, 3, 2>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tlv22, 2, 2>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv2x, 2, Eigen::Dynamic>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv2x, Eigen::Dynamic, 2>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlvx2, Eigen::Dynamic, 2>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlvx2, 2, Eigen::Dynamic>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlvxx, 2, 2>, TriangleType::lower>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvxx, Eigen::Dynamic, 2>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tuv22, 2, 2>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv2x, 2, Eigen::Dynamic>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv2x, Eigen::Dynamic, 2>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuvx2, Eigen::Dynamic, 2>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuvx2, 2, Eigen::Dynamic>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuvxx, 2, 2>, TriangleType::upper>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvxx, Eigen::Dynamic, 2>>);

  static_assert(not hermitian_adapter<Eigen::Reshaped<Z22, 2, 2>>);
  static_assert(not hermitian_adapter<Eigen::Reshaped<C22_2, 2, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Z22, 2, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C22_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, 4, 1>, Likelihood::maybe>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, 1, 4>, Likelihood::maybe>);
}
#endif // EIGEN_VERSION_AT_LEAST(3,4,0)


TEST(eigen3, Eigen_Reverse)
{
  static_assert(index_dimension_of_v<Eigen::Reverse<M2x, Eigen::Vertical>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<Mx2, Eigen::Vertical>, 1> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<M2x, Eigen::Horizontal>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<Mx2, Eigen::BothDirections>, 1> == 2);

  static_assert(one_dimensional<Eigen::Reverse<M11, Eigen::Vertical>>);
  static_assert(one_dimensional<Eigen::Reverse<M1x, Eigen::Horizontal>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Reverse<Mx1, Eigen::BothDirections>, Likelihood::maybe>);
  static_assert(one_dimensional<Eigen::Reverse<Mxx, Eigen::Vertical>, Likelihood::maybe>);

  static_assert(square_shaped<Eigen::Reverse<M22, Eigen::BothDirections>>);
  static_assert(square_shaped<Eigen::Reverse<M2x, Eigen::BothDirections>, Likelihood::maybe>);
  static_assert(square_shaped<Eigen::Reverse<Mx2, Eigen::BothDirections>, Likelihood::maybe>);
  static_assert(square_shaped<Eigen::Reverse<Mxx, Eigen::BothDirections>, Likelihood::maybe>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().reverse())> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<Z22, Eigen::Vertical>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<M11::IdentityReturnType, Eigen::Horizontal>> == 1);

  static_assert(zero<decltype(std::declval<Z23>().reverse())>);

  static_assert(identity_matrix<Eigen::Reverse<I22, Eigen::BothDirections>>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().reverse())>);
  static_assert(diagonal_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cxx_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().reverse()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv2x>().reverse()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tuvx2>().reverse()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tuvxx>().reverse()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Vertical>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Horizontal>, TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().reverse()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv2x>().reverse()), TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tlvx2>().reverse()), TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tlvxx>().reverse()), TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Vertical>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Horizontal>, TriangleType::upper>);
}


TEST(eigen3, Eigen_Select)
{
  auto br = make_eigen_matrix<bool, 2, 2>(true, false, true, false);
  auto bsa = eigen_matrix_t<bool, 2, 2>::Identity();

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_coefficient_v<decltype(br.select(std::declval<C22_2>(), std::declval<C22_2>()))> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd22_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd22_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<Cd22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(br.select(std::declval<Cd22_2>(), std::declval<Cd22_2>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B11_true>().select(std::declval<C11_2>(), std::declval<Z22>()))> == 2);

  static_assert(zero<decltype(std::declval<B22_true>().select(std::declval<Z22>(), M22::Identity()))>);
  static_assert(not zero<decltype(std::declval<B22_true>().select(std::declval<I22>(), std::declval<Z22>()))>);
  static_assert(zero<decltype(std::declval<B22_false>().select(std::declval<I22>(), std::declval<Z22>()))>);
  static_assert(zero<decltype(br.select(std::declval<Z22>(), std::declval<Z22>()))>);

  static_assert(diagonal_matrix<decltype(std::declval<B22_true>().select(std::declval<DW21>(), std::declval<M22>()))>);
  static_assert(diagonal_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<DW21>()))>);

  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Salv22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Sauv22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Salv22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Salv22>(), std::declval<Salv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sauv22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Salv22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sauv22>(), std::declval<Salv22>()))>);

  static_assert(triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tlv22>(), std::declval<M22>())), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tlv22>())), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tuv22>(), std::declval<M22>())), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tuv22>())), TriangleType::upper>);
}


TEST(eigen3, Eigen_SelfAdjointView)
{
  static_assert(std::is_same_v<nested_object_of_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>, M22&>);

  static_assert(not eigen_matrix_general<Eigen::SelfAdjointView<M33, Eigen::Lower>, true>);
  static_assert(not writable<Eigen::SelfAdjointView<M33, Eigen::Lower>>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M33, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M3x, Eigen::Lower>>, M3x>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<Mx3, Eigen::Lower>>, Mx3>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<Mxx, Eigen::Lower>>, Mxx>);

  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>>);
  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M22, Eigen::Upper>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<Mx2, Eigen::Lower>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<Mx2, Eigen::Upper>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M2x, Eigen::Lower>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M2x, Eigen::Upper>>>);

  static_assert(constant_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C22_2>, Eigen::Upper>> == 2);

  static_assert(constant_matrix<C11_1cx>);
  static_assert(std::real(constant_coefficient_v<C11_1cx>) == 1);
  static_assert(std::imag(constant_coefficient_v<C11_1cx>) == 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_1cx>, Eigen::Lower>>);

  static_assert(constant_matrix<C11_2cx>);
  EXPECT_EQ(std::real(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 2);
  EXPECT_EQ(std::imag(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_2cx>, Eigen::Lower>>);

  static_assert(constant_diagonal_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Upper>> == 2);

  static_assert(zero<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Z22>, Eigen::Upper>>);

  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M3x::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(Mx3::Identity(3, 3)), Eigen::Upper>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(Mxx::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);

  static_assert(diagonal_matrix<Sadv22>);
  static_assert(diagonal_matrix<Sadv2x, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadvx2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadvxx, Likelihood::maybe>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Lower>, HermitianAdapterType::lower>); // the diagonal must be real

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>, HermitianAdapterType::upper>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Upper>, HermitianAdapterType::upper>); // the diagonal must be real

  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<M43, Eigen::Lower>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<M34, Eigen::Upper>, HermitianAdapterType::upper>);

  auto m22l = make_eigen_matrix<double, 2, 2>(9, 0, 3, 10);
  auto salv22 = m22l.template selfadjointView<Eigen::Lower>();
  EXPECT_EQ(get_component(salv22, 0, 0), 9);
  EXPECT_EQ(get_component(salv22, 0, 1), 3);
  EXPECT_EQ(get_component(salv22, 1, 0), 3);
  EXPECT_EQ(get_component(salv22, 1, 1), 10);
  static_assert(std::is_lvalue_reference_v<decltype(get_component(salv22, 0, 1))>);
  static_assert(element_settable<decltype(salv22), 2>);
  set_component(salv22, 4, 0, 1);
  EXPECT_EQ(get_component(salv22, 0, 1), 4);
  EXPECT_EQ(get_component(salv22, 1, 0), 4);

  auto m22u = make_eigen_matrix<double, 2, 2>(9, 3, 0, 10);
  auto sauv22 = m22u.template selfadjointView<Eigen::Upper>();
  EXPECT_EQ(get_component(sauv22, 0, 0), 9);
  EXPECT_EQ(get_component(sauv22, 0, 1), 3);
  EXPECT_EQ(get_component(sauv22, 1, 0), 3);
  EXPECT_EQ(get_component(sauv22, 1, 1), 10);
  static_assert(std::is_lvalue_reference_v<decltype(get_component(sauv22, 1, 0))>);
  static_assert(element_settable<decltype(sauv22), 2>);
  set_component(sauv22, 4, 1, 0);
  EXPECT_EQ(get_component(sauv22, 0, 1), 4);
  EXPECT_EQ(get_component(sauv22, 1, 0), 4);

  auto m22lc = make_eigen_matrix<std::complex<double>, 2, 2>(std::complex<double>{9, 0.9}, 0, std::complex<double>{3, 0.3}, 10);
  auto salv22c = m22lc.template selfadjointView<Eigen::Lower>();
  EXPECT_EQ(std::real(get_component(salv22c, 0, 0)), 9);
  EXPECT_EQ(std::real(get_component(salv22c, 0, 1)), 3);
  EXPECT_EQ(std::imag(get_component(salv22c, 0, 1)), -0.3);
  EXPECT_EQ(std::real(get_component(salv22c, 1, 0)), 3);
  EXPECT_EQ(std::imag(get_component(salv22c, 1, 0)), 0.3);
  EXPECT_EQ(std::real(get_component(salv22c, 1, 1)), 10);
  static_assert(not std::is_lvalue_reference_v<decltype(get_component(salv22c, 0, 1))>);
  static_assert(element_settable<decltype(salv22c), 2>);
  set_component(salv22c, std::complex<double>{4, 0.4}, 0, 1);
  EXPECT_EQ(std::imag(get_component(salv22c, 0, 1)), 0.4);
  EXPECT_EQ(std::imag(get_component(salv22c, 1, 0)), -0.4);
}


TEST(eigen3, Eigen_Solve)
{
  static_assert(not self_contained<Eigen::Solve<Eigen::PartialPivLU<M31>, M31>>);
}


TEST(eigen3, Eigen_Transpose)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().transpose())> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().transpose())> == 2);

  static_assert(zero<decltype((std::declval<Z23>()).transpose())>);

  static_assert(identity_matrix<Eigen::Transpose<I22>>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().transpose())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().transpose()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().transpose()), TriangleType::upper>);
}


TEST(eigen3, Eigen_TriangularView)
{
  static_assert(std::is_same_v<nested_object_of_t<Eigen::TriangularView<M22, Eigen::Upper>>, M22&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M33, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M3x, Eigen::Upper>>, M3x>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<Mx3, Eigen::Upper>>, Mx3>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<Mxx, Eigen::Upper>>, Mxx>);

  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<C11_2>, Eigen::Lower>> == 2);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::Lower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Zxx>, Eigen::Upper>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<decltype(M32::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd2x_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cdx2_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cdxx_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::UnitLower>> == 1);

  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Lower>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z2x>, Eigen::StrictlyLower>> == 0);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::StrictlyLower>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Zx2>, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M2x::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mx2::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mxx::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<decltype(M23::Identity()), Eigen::StrictlyLower>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M2x::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mx2::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mxx::Identity()), Eigen::UnitLower>> == 1);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<decltype(M23::Identity()), Eigen::UnitUpper>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Tlv22, Eigen::UnitUpper>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Tuv22, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::UnitLower>> == 1);

  static_assert(zero<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>>);

  static_assert(identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M3x::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(Mx3::Identity(3, 3)), Eigen::Upper>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(Mxx::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::StrictlyUpper>>);
  static_assert(identity_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::UnitUpper>>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::TriangularView<M3x, Eigen::Lower>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mx3, Eigen::Lower>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mxx, Eigen::Lower>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Tlvx2, TriangleType::lower, Likelihood::maybe>);
  static_assert(not triangular_matrix<Eigen::TriangularView<M43, Eigen::Lower>, TriangleType::lower>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::TriangularView<M3x, Eigen::Upper>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mx3, Eigen::Upper>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mxx, Eigen::Upper>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Tuv2x, TriangleType::upper, Likelihood::maybe>);
  static_assert(not triangular_matrix<Eigen::TriangularView<M34, Eigen::Upper>, TriangleType::upper>);

  static_assert(triangular_matrix<Eigen::TriangularView<Tlv22, Eigen::Upper>, TriangleType::diagonal>);
  static_assert(triangular_matrix<Eigen::TriangularView<Tuv22, Eigen::Lower>, TriangleType::diagonal>);

  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv22, Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv2x, Eigen::Upper>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlvx2, Eigen::Upper>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlvxx, Eigen::Upper>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv22, Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv2x, Eigen::Lower>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuvx2, Eigen::Lower>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuvxx, Eigen::Lower>, Likelihood::maybe>);

  static_assert(hermitian_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Upper>>);

  auto m22l = make_eigen_matrix<double, 2, 2>(3, 1, 1, 3);
  auto tlv22 = m22l.template triangularView<Eigen::Lower>();
  EXPECT_EQ(get_component(tlv22, 0, 0), 3);
  EXPECT_EQ(get_component(tlv22, 0, 1), 0);
  EXPECT_EQ(get_component(tlv22, 1, 0), 1);
  EXPECT_EQ(get_component(tlv22, 1, 1), 3);
  static_assert(not element_settable<decltype(tlv22), 2>);
  static_assert(element_settable<decltype(nested_object(tlv22)), 2>);
  set_component(nested_object(tlv22), 4, 1, 0);
  EXPECT_EQ(get_component(tlv22, 1, 0), 4);
  EXPECT_EQ(get_component(tlv22, 0, 1), 0);

  auto m22u = make_eigen_matrix<double, 2, 2>(3, 1, 1, 3);
  auto tuv22 = m22u.template triangularView<Eigen::Upper>();
  EXPECT_EQ(get_component(tuv22, 0, 0), 3);
  EXPECT_EQ(get_component(tuv22, 0, 1), 1);
  EXPECT_EQ(get_component(tuv22, 1, 0), 0);
  EXPECT_EQ(get_component(tuv22, 1, 1), 3);
  static_assert(not element_settable<decltype(tuv22), 2>);
  static_assert(element_settable<decltype(nested_object(tuv22)), 2>);
  set_component(nested_object(tuv22), 4, 0, 1);
  EXPECT_EQ(get_component(tuv22, 0, 1), 4);
  EXPECT_EQ(get_component(tuv22, 1, 0), 0);

  auto m22lc = make_eigen_matrix<std::complex<double>, 2, 2>(std::complex<double>{3, 0.3}, 0, std::complex<double>{1, 0.1}, 3);
  auto tlv22c = m22lc.template triangularView<Eigen::Lower>();
  EXPECT_EQ(std::real(get_component(tlv22c, 0, 0)), 3);
  EXPECT_EQ(std::real(get_component(tlv22c, 0, 1)), 0);
  EXPECT_EQ(std::imag(get_component(tlv22c, 0, 1)), 0);
  EXPECT_EQ(std::real(get_component(tlv22c, 1, 0)), 1);
  EXPECT_EQ(std::imag(get_component(tlv22c, 1, 0)), 0.1);
  EXPECT_EQ(std::real(get_component(tlv22c, 1, 1)), 3);
  static_assert(not element_settable<decltype(tlv22c), 2>);
  static_assert(element_settable<decltype(nested_object(tlv22c)), 2>);
  set_component(nested_object(tlv22c), std::complex<double>{4, 0.4}, 1, 0);
  EXPECT_EQ(std::imag(get_component(tlv22c, 1, 0)), 0.4);
  EXPECT_EQ(std::imag(get_component(tlv22c, 0, 1)), 0);
}


TEST(eigen3, Eigen_VectorBlock)
{
  static_assert(eigen_matrix_general<Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 1>, true>);
  static_assert(eigen_array_general<Eigen::VectorBlock<Eigen::Array<double, 2, 1>, 1>, true>);
  static_assert(std::is_same_v<double, typename Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 0>::Scalar>);

  static_assert(eigen_general<decltype(std::declval<C21_2>().segment<1>(0)), true>);

  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().segment<1>(0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_m2>().segment(1, 0))> == -2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().segment<1>(0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_m2>().segment<1>(0))> == -2);

  static_assert(zero<decltype(std::declval<Z21>().segment<1>(0))>);
  static_assert(zero<decltype(std::declval<Z21>().segment(1, 0))>);

}
