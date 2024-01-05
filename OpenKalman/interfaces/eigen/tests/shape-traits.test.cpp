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


TEST(eigen3, shape_traits)
{
  static_assert(one_dimensional<I11, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Zx1, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Zx1>);
  static_assert(one_dimensional<C11_1>);
  static_assert(one_dimensional<C1x_1, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<C1x_1>);
  static_assert(one_dimensional<Cxx_1, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Cxx_1>);
  static_assert(one_dimensional<C11_m1>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<M11>>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M1x>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<M1x>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<Mx1>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mxx>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<Mxx>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M22>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M2x>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mx2>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Cd22_2, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Cd2x_2, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<EigenWrapper<Eigen::DiagonalWrapper<Cx1_1>>, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<EigenWrapper<Eigen::DiagonalWrapper<C1x_1>>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<M22>);
  static_assert(square_shaped<M2x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Mx2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Mxx, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<M11>);
  static_assert(square_shaped<M1x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Mx1, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<C11_m1, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Z22, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Z2x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Zx2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Zxx, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<C22_2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<C2x_2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Cx2_2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Cxx_2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<DM2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<DMx, Qualification::depends_on_dynamic_shape>);
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
  static_assert(square_shaped<Tlv2x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Tlvx2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Tlvxx, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Tlv2x>);
  static_assert(not square_shaped<Tlvx2>);
  static_assert(not square_shaped<Tlvxx>);

  static_assert(square_shaped<Salv22>);
  static_assert(square_shaped<Salv2x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Salvx2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Salvxx, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Salv2x>);
  static_assert(not square_shaped<Salvx2>);
  static_assert(not square_shaped<Salvxx>);

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

