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
   * \details A triangular matrix need not be \ref square_shaped, but it must be zero either above or below the diagonal (or both).
   * The vector space descriptors are not taken into consideration, so they do not necessarily need to match.
   * \note One-dimensional matrices or vectors are considered to be triangular, and a vector is triangular if
   * every component other than its first component is zero.
   * \tparam T A matrix or tensor.
   * \tparam t The \ref TriangleType
   * triangular if it is \ref one-dimensional, and that is not necessarily known at compile time.
   */
  template<typename T, TriangleType t = TriangleType::any>
#ifdef __cpp_concepts
  concept triangular_matrix = indexible<T> and
    (interface::indexible_object_traits<std::decay_t<T>>::template is_triangular<t> or
      constant_diagonal_matrix<T, ConstantType::any> or zero<T>);
#else
  constexpr bool triangular_matrix = interface::is_explicitly_triangular<T, t>::value or
    constant_diagonal_matrix<T, ConstantType::any> or zero<T>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_TRIANGULAR_MATRIX_HPP
