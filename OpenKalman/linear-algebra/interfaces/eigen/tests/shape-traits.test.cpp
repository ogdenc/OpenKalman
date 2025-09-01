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

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, shape_traits)
{
  static_assert(one_dimensional<I11, applicability::permitted>);
  static_assert(one_dimensional<Zx1, applicability::permitted>);
  static_assert(not one_dimensional<Zx1>);
  static_assert(one_dimensional<C11_1>);
  static_assert(one_dimensional<C1x_1, applicability::permitted>);
  static_assert(not one_dimensional<C1x_1>);
  static_assert(one_dimensional<Cxx_1, applicability::permitted>);
  static_assert(not one_dimensional<Cxx_1>);
  static_assert(one_dimensional<C11_m1>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<M11>>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M1x>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<M1x>, applicability::permitted>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mx1>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<Mx1>, applicability::permitted>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mxx>>);
  static_assert(one_dimensional<Eigen::DiagonalWrapper<Mxx>, applicability::permitted>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M22>, applicability::permitted>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<M2x>, applicability::permitted>);
  static_assert(not one_dimensional<Eigen::DiagonalWrapper<Mx2>, applicability::permitted>);
  static_assert(not one_dimensional<Cd22_2, applicability::permitted>);
  static_assert(not one_dimensional<Cd2x_2, applicability::permitted>);
  static_assert(one_dimensional<EigenWrapper<Eigen::DiagonalWrapper<Cx1_1>>, applicability::permitted>);
  static_assert(one_dimensional<EigenWrapper<Eigen::DiagonalWrapper<C1x_1>>, applicability::permitted>);

  static_assert(square_shaped<M22>);
  static_assert(square_shaped<M2x, applicability::permitted>);
  static_assert(square_shaped<Mx2, applicability::permitted>);
  static_assert(square_shaped<Mxx, applicability::permitted>);
  static_assert(square_shaped<M11>);
  static_assert(square_shaped<M1x, applicability::permitted>);
  static_assert(square_shaped<Mx1, applicability::permitted>);

  static_assert(square_shaped<C11_m1, applicability::permitted>);
  static_assert(square_shaped<Z22, applicability::permitted>);
  static_assert(square_shaped<Z2x, applicability::permitted>);
  static_assert(square_shaped<Zx2, applicability::permitted>);
  static_assert(square_shaped<Zxx, applicability::permitted>);
  static_assert(square_shaped<C22_2, applicability::permitted>);
  static_assert(square_shaped<C2x_2, applicability::permitted>);
  static_assert(square_shaped<Cx2_2, applicability::permitted>);
  static_assert(square_shaped<Cxx_2, applicability::permitted>);
  static_assert(square_shaped<DM2, applicability::permitted>);
  static_assert(square_shaped<DMx, applicability::permitted>);
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
  static_assert(square_shaped<Tlv2x, applicability::permitted>);
  static_assert(square_shaped<Tlvx2, applicability::permitted>);
  static_assert(square_shaped<Tlvxx, applicability::permitted>);
  static_assert(not square_shaped<Tlv2x>);
  static_assert(not square_shaped<Tlvx2>);
  static_assert(not square_shaped<Tlvxx>);

  static_assert(square_shaped<Salv22>);
  static_assert(square_shaped<Salv2x, applicability::permitted>);
  static_assert(square_shaped<Salvx2, applicability::permitted>);
  static_assert(square_shaped<Salvxx, applicability::permitted>);
  static_assert(not square_shaped<Salv2x>);
  static_assert(not square_shaped<Salvx2>);
  static_assert(not square_shaped<Salvxx>);

  static_assert(vector_space_descriptors_may_match_with<>);
  static_assert(vector_space_descriptors_may_match_with<M32>);
  static_assert(vector_space_descriptors_may_match_with<M32, Mx2, M3x>);
  static_assert(vector_space_descriptors_may_match_with<M2x, M23, Mx3>);
  static_assert(vector_space_descriptors_may_match_with<M2x, Z23>);
  static_assert(vector_space_descriptors_may_match_with<M2x, Mx3>);

  static_assert(vector_space_descriptors_match_with<M32, M32>);
  static_assert(not vector_space_descriptors_match_with<M32, Mx2>);
  static_assert(not vector_space_descriptors_match_with<Mx2, M32>);
  static_assert(vector_space_descriptors_match_with<M22, Salv22, M22, Z22>);
}

