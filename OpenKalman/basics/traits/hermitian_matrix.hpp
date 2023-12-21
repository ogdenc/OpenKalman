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
 * \brief Definition for \ref hermitian_matrix.
 */

#ifndef OPENKALMAN_HERMITIAN_MATRIX_HPP
#define OPENKALMAN_HERMITIAN_MATRIX_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_inferred_hermitian_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_inferred_hermitian_matrix<T, b, std::enable_if_t<not complex_number<typename scalar_type_of<T>::type> or
      zero<T> or real_axis_number<constant_coefficient<T>> or real_axis_number<constant_diagonal_coefficient<T>>>>
      : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix (assuming it is a \ref square_shaped).
   * \tparam T A matrix or tensor.
   * \tparam b Whether T must be known to be a square matrix at compile time.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept hermitian_matrix = indexible<T> and
    ((interface::indexible_object_traits<std::decay_t<T>>::is_hermitian and square_shaped<T, b>) or
      (((constant_matrix<T, CompileTimeStatus::any, b> and square_shaped<T, b>) or diagonal_matrix<T, b>) and
      (not complex_number<scalar_type_of_t<T>> or zero<T> or
          real_axis_number<constant_coefficient<T>> or real_axis_number<constant_diagonal_coefficient<T>>)));
#else
  constexpr bool hermitian_matrix = (interface::is_explicitly_hermitian<T>::value and square_shaped<T, b>) or
    (((constant_matrix<T, CompileTimeStatus::any, b> and square_shaped<T, b>) or diagonal_matrix<T, b>) and
      detail::is_inferred_hermitian_matrix<T, b>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_HERMITIAN_MATRIX_HPP
