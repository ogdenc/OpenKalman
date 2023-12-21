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
 * \brief Definition for \ref identity_matrix.
 */

#ifndef OPENKALMAN_IDENTITY_MATRIX_HPP
#define OPENKALMAN_IDENTITY_MATRIX_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_identity_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_identity_matrix<T, b, std::enable_if_t<constant_diagonal_matrix<T, CompileTimeStatus::known, b>>>
      : std::bool_constant<internal::are_within_tolerance(constant_diagonal_coefficient_v<T>, 1)> {};
  }
#endif

  /**
   * \brief Specifies that a type is an identity matrix.
   * \details A zero-dimensional matrix is also an identity matrix.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == Likelihood::definitely</code>: T is known at compile time to be identity; or
   * - if <code>b == Likelihood::maybe</code>: either
   * -- it is known at compile time that T <em>may</em> be a \ref constant_diagonal_matrix and that its value is 1; or
   * -- it is unknown at compile time whether T is a zero vector (i.e., a zero-dimensional object).
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept identity_matrix =
    (constant_diagonal_matrix<T, CompileTimeStatus::known, b> and internal::are_within_tolerance(constant_diagonal_coefficient_v<T>, 1)) or
#else
  constexpr bool identity_matrix = detail::is_identity_matrix<T, b>::value or
#endif
    (index_count_v<T> == 2 and dimension_size_of_index_is<T, 0, 0, b> and dimension_size_of_index_is<T, 1, 0, b>);


} // namespace OpenKalman

#endif //OPENKALMAN_IDENTITY_MATRIX_HPP
