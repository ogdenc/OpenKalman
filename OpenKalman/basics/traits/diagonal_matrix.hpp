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
 * \brief Definition for \ref diagonal_matrix.
 */

#ifndef OPENKALMAN_DIAGONAL_MATRIX_HPP
#define OPENKALMAN_DIAGONAL_MATRIX_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that a type is a diagonal matrix.
   * \details A diagonal matrix has zero components everywhere except the main diagonal. It is not necessarily a
   * square matrix.
   * \note A \ref diagonal_adapter is an unqualified diagonal matrix, but not all diagonal matrices are diagonal adapters.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept diagonal_matrix =
#else
  constexpr bool diagonal_matrix =
#endif
    triangular_matrix<T, TriangleType::diagonal>;


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_MATRIX_HPP
