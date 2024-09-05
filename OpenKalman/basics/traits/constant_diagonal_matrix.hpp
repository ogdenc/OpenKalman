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
 * \brief Definition for \ref constant_diagonal_matrix.
 */

#ifndef OPENKALMAN_CONSTANT_DIAGONAL_MATRIX_HPP
#define OPENKALMAN_CONSTANT_DIAGONAL_MATRIX_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that all diagonal elements of a diagonal object are the same constant value.
   * \details A constant diagonal matrix is also a \ref diagonal_matrix. It is not necessarily square.
   * If T is a rank >2 tensor, every rank-2 slice comprising dimensions 0 and 1 must be constant diagonal matrix.
   */
  template<typename T, ConstantType c = ConstantType::any>
#ifdef __cpp_concepts
  concept constant_diagonal_matrix =
#else
  constexpr bool constant_diagonal_matrix =
#endif
    indexible<T> and scalar_constant<constant_diagonal_coefficient<T>, c>;


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_DIAGONAL_MATRIX_HPP
