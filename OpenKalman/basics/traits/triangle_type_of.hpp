/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref triangle_type_of.
 */

#ifndef OPENKALMAN_TRIANGLE_TYPE_OF_HPP
#define OPENKALMAN_TRIANGLE_TYPE_OF_HPP


namespace OpenKalman
{
  /**
   * \brief The common \ref TriangleType associated with a set of \ref triangular_matrix "triangular matrices".
   * \details If there is no common triangle type, the result is TriangleType::any.
   * The result here does not guarantee that any of the parameters are unqualified triangular, which must be checked
   * with \ref triangular_matrix.
   */
  template<typename T, typename...Ts>
  struct triangle_type_of
    : std::integral_constant<TriangleType,
      (triangular_matrix<T, TriangleType::diagonal, Qualification::depends_on_dynamic_shape> and ... and triangular_matrix<Ts, TriangleType::diagonal, Qualification::depends_on_dynamic_shape>) ? TriangleType::diagonal :
      (triangular_matrix<T, TriangleType::lower, Qualification::depends_on_dynamic_shape> and ... and triangular_matrix<Ts, TriangleType::lower, Qualification::depends_on_dynamic_shape>) ? TriangleType::lower :
      (triangular_matrix<T, TriangleType::upper, Qualification::depends_on_dynamic_shape> and ... and triangular_matrix<Ts, TriangleType::upper, Qualification::depends_on_dynamic_shape>) ? TriangleType::upper :
      TriangleType::any> {};


  /**
   * \brief The TriangleType associated with a \ref triangular_matrix.
   */
  template<typename T, typename...Ts>
  constexpr auto triangle_type_of_v = triangle_type_of<T, Ts...>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_TRIANGLE_TYPE_OF_HPP
