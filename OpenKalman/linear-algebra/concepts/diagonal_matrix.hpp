/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "linear-algebra/concepts/triangular_matrix.hpp"

namespace OpenKalman
{
  /**
   * \brief Specifies that a type is a diagonal matrix or tensor.
   * \details A diagonal matrix has zero components everywhere except the main diagonal. It is not necessarily square.
   * For rank >2 tensors, every rank-2 slice comprising dimensions 0 and 1 must be diagonal.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept diagonal_matrix =
#else
  constexpr bool diagonal_matrix =
#endif
    triangular_matrix<T, triangle_type::diagonal>;

}

#endif
