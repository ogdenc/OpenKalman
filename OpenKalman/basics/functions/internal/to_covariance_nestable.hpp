/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref to_covariance_nestable function.
 */

#ifndef OPENKALMAN_TO_COVARIANCE_NESTABLE_HPP
#define OPENKALMAN_TO_COVARIANCE_NESTABLE_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Convert a \ref covariance_nestable matrix or \ref typed_matrix_nestable to a \ref covariance_nestable.
   * \tparam T \ref covariance_nestable to which Arg is to be converted.
   * \return A \ref covariance_nestable of type T.
   */
#ifdef __cpp_concepts
  template<covariance_nestable T, typename Arg> requires
    (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_shaped<Arg> or vector<Arg>))) and
    (index_dimension_of_v<Arg, 0> == index_dimension_of_v<T, 0>) and
    (not zero<T> or zero<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or vector<Arg>)
#else
  template<typename T, typename Arg, typename = std::enable_if_t<
    (not std::is_same_v<T, Arg>) and covariance_nestable<T> and
    (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_shaped<Arg> or vector<Arg>))) and
    (index_dimension_of<Arg, 0>::value == index_dimension_of<T, 0>::value) and
    (not zero<T> or zero<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or vector<Arg>)>>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&&) noexcept;


  /**
   * \overload
   * \internal
   * \brief Convert \ref covariance or \ref typed_matrix to a \ref covariance_nestable of type T.
   * \tparam T \ref covariance_nestable to which Arg is to be converted.
   * \tparam Arg A \ref covariance or \ref typed_matrix.
   * \return A \ref covariance_nestable of type T.
   */
#ifdef __cpp_concepts
  template<covariance_nestable T, typename Arg> requires
    (covariance<Arg> or (typed_matrix<Arg> and (square_shaped<Arg> or vector<Arg>))) and
    (index_dimension_of_v<Arg, 0> == index_dimension_of_v<T, 0>) and
    (not zero<T> or zero<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or vector<Arg>)
#else
  template<typename T, typename Arg, typename = void, typename = std::enable_if_t<
    (not std::is_same_v<T, Arg>) and covariance_nestable<T> and (not std::is_void_v<Arg>) and
    (covariance<Arg> or (typed_matrix<Arg> and (square_shaped<Arg> or vector<Arg>))) and
    (index_dimension_of<Arg, 0>::value == index_dimension_of<T, 0>::value) and
    (not zero<T> or zero<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or vector<Arg>)>>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&&) noexcept;


  /**
   * \overload
   * \internal
   * /return The result of converting Arg to a \ref covariance_nestable.
   */
#ifdef __cpp_concepts
  template<typename Arg>
  requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_shaped<Arg> or vector<Arg>))
#else
  template<typename Arg, typename = std::enable_if_t<covariance_nestable<Arg> or
      (typed_matrix_nestable<Arg> and (square_shaped<Arg> or vector<Arg>))>>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&&) noexcept;


  /**
   * \overload
   * \internal
   * /return A \ref triangular_matrix if Arg is a \ref triangular_covariance or otherwise a \ref hermitian_matrix.
   */
#ifdef __cpp_concepts
  template<typename Arg> requires covariance<Arg> or
    (typed_matrix<Arg> and (square_shaped<Arg> or vector<Arg>))
#else
  template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
    (typed_matrix<Arg> and (square_shaped<Arg> or vector<Arg>))>>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&&) noexcept;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_TO_COVARIANCE_NESTABLE_HPP
