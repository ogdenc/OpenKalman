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
  // ------------------------------- //
  //  constant_diagonal_coefficient  //
  // ------------------------------- //

  /**
   * \brief The constant associated with T, assuming T is a \ref constant_diagonal_matrix.
   * \details Before using this value, always check if T is a \ref constant_diagonal_matrix, because
   * the value may be defined in some cases where T is not actually a constant diagonal matrix.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct constant_diagonal_coefficient
  {
    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};
  };


  /**
   * \brief Deduction guide for \ref constant_diagonal_coefficient.
   */
  template<typename T>
  explicit constant_diagonal_coefficient(T&&) -> constant_diagonal_coefficient<std::decay_t<T>>;


  /// Helper template for constant_diagonal_coefficient.
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  constexpr auto constant_diagonal_coefficient_v = constant_diagonal_coefficient<T>::value;


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_zero : std::false_type {};

    template<typename T>
    struct is_zero<T, std::enable_if_t<constant_matrix<T, ConstantType::static_constant>>>
      : std::bool_constant<internal::are_within_tolerance(constant_coefficient_v<T>, 0)> {};
  }
#endif


  // -------------------------- //
  //  constant_diagonal_matrix  //
  // -------------------------- //

  /**
   * \brief Specifies that all diagonal elements of a diagonal object are the same constant value.
   */
  template<typename T, ConstantType c = ConstantType::any>
#ifdef __cpp_concepts
  concept constant_diagonal_matrix = indexible<T> and
    (scalar_constant<constant_diagonal_coefficient<T>, c> or
      (constant_matrix<T, ConstantType::static_constant> and internal::are_within_tolerance(constant_coefficient_v<T>, 0)) or
      (c != ConstantType::static_constant and one_dimensional<T>));
#else
  constexpr bool constant_diagonal_matrix =
    (scalar_constant<constant_diagonal_coefficient<T>, c> or detail::is_zero<T>::value or
    (c != ConstantType::static_constant and one_dimensional<T>));
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_DIAGONAL_MATRIX_HPP
