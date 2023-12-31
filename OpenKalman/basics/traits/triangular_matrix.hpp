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
 * \brief Definition for \ref triangular_matrix.
 */

#ifndef OPENKALMAN_TRIANGULAR_MATRIX_HPP
#define OPENKALMAN_TRIANGULAR_MATRIX_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that a type is a triangular matrix (upper, lower, or diagonal).
   * \tparam T A matrix or tensor.
   */
  template<typename T, TriangleType t = TriangleType::any, Qualification b = Qualification::unqualified>
#ifdef __cpp_concepts
  concept triangular_matrix = indexible<T> and
    ((interface::indexible_object_traits<std::decay_t<T>>::template is_triangular<t, square_shaped<T> ? Qualification::depends_on_dynamic_shape : b> and square_shaped<T, b>) or
    constant_diagonal_matrix<T, ConstantType::any, b>);
#else
  constexpr bool triangular_matrix =
    ((interface::is_explicitly_triangular<T, t, square_shaped<T> ? Qualification::depends_on_dynamic_shape : b>::value and square_shaped<T, b>) or
    constant_diagonal_matrix<T, ConstantType::any, b>);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_TRIANGULAR_MATRIX_HPP
