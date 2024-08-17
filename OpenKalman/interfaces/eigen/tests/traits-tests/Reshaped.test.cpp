/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_Reshaped)
{
  static_assert(eigen_matrix_general<Eigen::Reshaped<M32, 2, 3, Eigen::RowMajor>, true>);
  static_assert(eigen_array_general<Eigen::Reshaped<Eigen::ArrayWrapper<M32>, 2, 3, Eigen::RowMajor>, true>);
  auto m32 = make_eigen_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6);
  auto m23 = make_eigen_matrix<double, 2, 3>(1, 4, 2, 5, 3, 6);
  EXPECT_TRUE(is_near(m32.reshaped<Eigen::RowMajor>(2, 3), m23));

  static_assert(index_dimension_of_v<Eigen::Reshaped<Mxx, 3, Eigen::Dynamic>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Mxx, Eigen::Dynamic, 4>, 1> == 4);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M21, Eigen::Dynamic, 2>, 0> == 1);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M21, 1, Eigen::Dynamic>, 1> == 2);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M12, Eigen::Dynamic, 1>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M12, 2, Eigen::Dynamic>, 1> == 1);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, 4>, 0> == 4);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, 4, Eigen::Dynamic>, 1> == 4);

  static_assert(one_dimensional<Eigen::Reshaped<M11, 1, 1>>);
  static_assert(one_dimensional<Eigen::Reshaped<M11, 1, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<M11, Eigen::Dynamic, 1>>);
  static_assert(one_dimensional<Eigen::Reshaped<M11, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(one_dimensional<Eigen::Reshaped<M1x, 1, 1>>);
  static_assert(not one_dimensional<Eigen::Reshaped<M1x, 1, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<M1x, 1, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<M1x, Eigen::Dynamic, 1>>);
  static_assert(one_dimensional<Eigen::Reshaped<M1x, Eigen::Dynamic, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<M1x, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<M1x, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(one_dimensional<Eigen::Reshaped<Mx1, 1, 1>>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx1, 1, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mx1, 1, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx1, Eigen::Dynamic, 1>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mx1, Eigen::Dynamic, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx1, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mx1, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(one_dimensional<Eigen::Reshaped<Mxx, 1, 1>>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mxx, 1, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mxx, 1, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mxx, Eigen::Dynamic, 1>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mxx, Eigen::Dynamic, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(one_dimensional<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(not one_dimensional<Eigen::Reshaped<Mxx, 2, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<M2x, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<M2x, Eigen::Dynamic, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx2, 1, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<M2x, 1, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Reshaped<Mx2, 1, 1>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<Eigen::Reshaped<M11, 1, 1>>);
  static_assert(square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, 4, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, 4>>);
  static_assert(square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(not square_shaped<Eigen::Reshaped<Eigen::Matrix<double, 2, 9>, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M5x, 2, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx5, 2, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, 1, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, 1, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, 1, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M21, Eigen::Dynamic, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M21, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<Eigen::Reshaped<M22, 2, 2>>);
  static_assert(square_shaped<Eigen::Reshaped<M22, 2, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<M22, Eigen::Dynamic, 2>>);
  static_assert(not square_shaped<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<Eigen::Reshaped<M2x, 2, 2>>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, 2, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<M2x, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, 2>>);
  static_assert(square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<M2x, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<Eigen::Reshaped<Mx2, 2, 2>>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, 2, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Mx2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, Eigen::Dynamic, 2>>);
  static_assert(square_shaped<Eigen::Reshaped<Mx2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mx2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Mx2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<Eigen::Reshaped<Mxx, 2, 2>>);
  static_assert(not square_shaped<Eigen::Reshaped<Mxx, 2, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Mxx, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, 2>>);
  static_assert(square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Reshaped<Mxx, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

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
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd22_2, 2, Eigen::Dynamic>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<decltype(std::declval<I32>() + std::declval<I32>()), 3, 2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<decltype(std::declval<I32>() + std::declval<I32>()), 3, Eigen::Dynamic>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<decltype(std::declval<I32>() + std::declval<I32>()), 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<decltype(std::declval<I32>() + std::declval<I32>()), Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<decltype(std::declval<I32>() + std::declval<I32>()), Eigen::Dynamic, 3>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<decltype(std::declval<I32>() + std::declval<I32>()), Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd2x_2, 2, Eigen::Dynamic>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, 2>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdx2_2, 2, Eigen::Dynamic>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, 2, 2>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, 2, Eigen::Dynamic>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, 2>>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(zero<Eigen::Reshaped<Z22, 4, 1>>);
  static_assert(zero<Eigen::Reshaped<Z21, 1, 2>>);
  static_assert(zero<Eigen::Reshaped<Z23, 3, 2>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tlv22, 2, 2>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv22, 2, Eigen::Dynamic>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv22, Eigen::Dynamic, 2>, TriangleType::lower>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlv22, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tlv2x, 2, 2>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv2x, 2, Eigen::Dynamic>, TriangleType::lower>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlv2x, Eigen::Dynamic, 2>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlv2x, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tlvx2, 2, 2>, TriangleType::lower>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvx2, 2, Eigen::Dynamic>>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlvx2, Eigen::Dynamic, 2>, TriangleType::lower>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvx2, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvxx, 2, 2>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvxx, 2, Eigen::Dynamic>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvxx, Eigen::Dynamic, 2>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlvxx, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tuv2x, 2, 2>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv2x, 2, Eigen::Dynamic>, TriangleType::upper>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuv2x, Eigen::Dynamic, 2>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuv2x, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tuvx2, 2, 2>, TriangleType::upper>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvx2, 2, Eigen::Dynamic>>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuvx2, Eigen::Dynamic, 2>, TriangleType::upper>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvx2, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvxx, 2, 2>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvxx, 2, Eigen::Dynamic>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvxx, Eigen::Dynamic, 2>>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuvxx, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(not hermitian_adapter<Eigen::Reshaped<Z22, 2, 2>>);
  static_assert(not hermitian_adapter<Eigen::Reshaped<C22_2, 2, 2>>);

  static_assert(hermitian_matrix<Eigen::Reshaped<Z22, 2, 2>>);

  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, 4, 1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, 1, 4>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<C22_2, 2, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C22_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C22_2, Eigen::Dynamic, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C22_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<C2x_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C2x_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C2x_2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C2x_2, Eigen::Dynamic, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C2x_2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C2x_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C2x_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<Cx2_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cx2_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cx2_2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cx2_2, Eigen::Dynamic, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cx2_2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cx2_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cx2_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<Cxx_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cxx_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cxx_2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cxx_2, Eigen::Dynamic, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cxx_2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cxx_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cxx_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);


  static_assert(hermitian_matrix<Eigen::Reshaped<Cd22_2, 2, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cd22_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<Cd2x_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cd2x_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cd2x_2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cd2x_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<Cdx2_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cdx2_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cdx2_2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cdx2_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);

  static_assert(hermitian_matrix<Eigen::Reshaped<Cdxx_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cdxx_2, 2, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cdxx_2, 2, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, 2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Cdxx_2, Eigen::Dynamic, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
}
#endif // EIGEN_VERSION_AT_LEAST(3,4,0)

