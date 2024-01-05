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


TEST(eigen3, special_matrix_traits)
{
  static_assert(triangular_matrix<Z22, TriangleType::lower>);
  static_assert(triangular_matrix<Z2x, TriangleType::lower>);
  static_assert(triangular_matrix<Zx2, TriangleType::lower>);
  static_assert(triangular_matrix<Zxx, TriangleType::lower>);
  static_assert(triangular_matrix<C11_2, TriangleType::lower>);
  static_assert(not triangular_matrix<C22_2, TriangleType::lower>);
  static_assert(triangular_matrix<I22, TriangleType::lower>);
  static_assert(triangular_matrix<I2x, TriangleType::lower>);
  static_assert(triangular_matrix<Ixx, TriangleType::lower>);
  static_assert(triangular_matrix<Cd22_2, TriangleType::lower>);
  static_assert(triangular_matrix<Cd2x_2, TriangleType::lower>);
  static_assert(triangular_matrix<Cdx2_2, TriangleType::lower>);
  static_assert(triangular_matrix<Cdxx_2, TriangleType::lower>);
  static_assert(triangular_matrix<DW21, TriangleType::lower>);
  static_assert(triangular_matrix<DW2x, TriangleType::lower>);
  static_assert(triangular_matrix<DWx1, TriangleType::lower>);
  static_assert(triangular_matrix<DWxx, TriangleType::lower>);

  static_assert(triangular_matrix<Z22, TriangleType::upper>);
  static_assert(triangular_matrix<Z2x, TriangleType::upper>);
  static_assert(triangular_matrix<Zx2, TriangleType::upper>);
  static_assert(triangular_matrix<Zxx, TriangleType::upper>);
  static_assert(triangular_matrix<Z2x, TriangleType::upper>);
  static_assert(triangular_matrix<Zx2, TriangleType::upper>);
  static_assert(triangular_matrix<Zxx, TriangleType::upper>);
  static_assert(triangular_matrix<C11_2, TriangleType::upper>);
  static_assert(not triangular_matrix<C22_2, TriangleType::upper>);
  static_assert(not triangular_matrix<C22_2, TriangleType::upper>);

  static_assert(triangular_matrix<I22, TriangleType::upper>);
  static_assert(triangular_matrix<I2x, TriangleType::upper>);
  static_assert(triangular_matrix<Ixx, TriangleType::upper>);
  static_assert(triangular_matrix<Cd22_2, TriangleType::upper>);
  static_assert(triangular_matrix<Cd2x_2, TriangleType::upper>);
  static_assert(triangular_matrix<Cdx2_2, TriangleType::upper>);
  static_assert(triangular_matrix<Cdxx_2, TriangleType::upper>);
  static_assert(triangular_matrix<DW21, TriangleType::upper>);
  static_assert(triangular_matrix<DW2x, TriangleType::upper>);
  static_assert(triangular_matrix<DWx1, TriangleType::upper>);
  static_assert(triangular_matrix<DWxx, TriangleType::upper>);

  static_assert(triangular_matrix<Tlv22, TriangleType::lower>);
  static_assert(triangular_matrix<Tlv2x, TriangleType::lower>);
  static_assert(triangular_matrix<Tlvx2, TriangleType::lower>);
  static_assert(triangular_matrix<Tlvxx, TriangleType::lower>);
  static_assert(triangular_matrix<Tlv2x, TriangleType::lower>);
  static_assert(triangular_matrix<Tlvx2, TriangleType::lower>);
  static_assert(triangular_matrix<Tlvxx, TriangleType::lower>);
  static_assert(not triangular_matrix<Tuv22, TriangleType::lower>);

  static_assert(triangular_matrix<Tuv22, TriangleType::upper>);
  static_assert(triangular_matrix<Tuv2x, TriangleType::upper>);
  static_assert(triangular_matrix<Tuvx2, TriangleType::upper>);
  static_assert(triangular_matrix<Tuvxx, TriangleType::upper>);
  static_assert(triangular_matrix<Tuv2x, TriangleType::upper>);
  static_assert(triangular_matrix<Tuvx2, TriangleType::upper>);
  static_assert(triangular_matrix<Tuvxx, TriangleType::upper>);
  static_assert(not triangular_matrix<Tlv22, TriangleType::upper>);

  static_assert(diagonal_matrix<Z22>);
  static_assert(diagonal_matrix<Z22>);
  static_assert(diagonal_matrix<Z2x>);
  static_assert(diagonal_matrix<Zx2>);
  static_assert(diagonal_matrix<Zxx>);
  static_assert(diagonal_matrix<C11_2>);
  static_assert(diagonal_matrix<I22>);
  static_assert(diagonal_matrix<I2x>);
  static_assert(diagonal_matrix<Ixx>);
  static_assert(diagonal_matrix<Cd22_2>);
  static_assert(diagonal_matrix<Cd2x_2>);
  static_assert(diagonal_matrix<Cdx2_2>);
  static_assert(diagonal_matrix<Cdxx_2>);
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
  static_assert(diagonal_matrix<Sadv2x>);
  static_assert(diagonal_matrix<Sadvx2>);
  static_assert(diagonal_matrix<Sadvxx>);
  static_assert(diagonal_matrix<M11>);
  static_assert(not diagonal_matrix<M1x>);
  static_assert(not diagonal_matrix<Mx1>);
  static_assert(not diagonal_matrix<Mxx>);
  static_assert(diagonal_matrix<M11>);
  static_assert(not diagonal_matrix<M1x>);
  static_assert(not diagonal_matrix<Mx1>);
  static_assert(not diagonal_matrix<Mxx>);

  static_assert(not diagonal_adapter<M11>);
  static_assert(not diagonal_adapter<M1x>);
  static_assert(not diagonal_adapter<Mx1>);
  static_assert(not diagonal_adapter<Mxx>);
  static_assert(not diagonal_adapter<DM2>);
  static_assert(diagonal_adapter<Eigen::DiagonalMatrix<double, 2>>);
  static_assert(diagonal_adapter<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M2x>>);
  static_assert(not diagonal_adapter<DW2x>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M2x>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M1x>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(not diagonal_adapter<DWxx>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<Mxx>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M22>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M22>>);

  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<Z2x, Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<Zx2, Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<Zxx, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_adapter<Z22>);
  static_assert(not hermitian_adapter<Z2x>);
  static_assert(not hermitian_adapter<Zx2>);
  static_assert(not hermitian_adapter<Zxx>);
  static_assert(hermitian_matrix<C22_2>);
  static_assert(hermitian_matrix<I22>);
  static_assert(hermitian_matrix<I2x, Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<Cd22_2>);
  static_assert(hermitian_matrix<Cd2x_2, Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<Cdx2_2, Qualification::depends_on_dynamic_shape>);
  static_assert(hermitian_matrix<Cdxx_2, Qualification::depends_on_dynamic_shape>);
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
}

