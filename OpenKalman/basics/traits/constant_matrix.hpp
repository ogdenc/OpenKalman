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
 * \brief Definition for \ref constant_matrix.
 */

#ifndef OPENKALMAN_CONSTANT_MATRIX_HPP
#define OPENKALMAN_CONSTANT_MATRIX_HPP


namespace OpenKalman
{
  // ---------------------- //
  //  constant_coefficient  //
  // ---------------------- //

  /**
   * \brief The constant associated with T, assuming T is a \ref constant_matrix.
   * \details Before using this value, always check if T is a \ref constant_matrix, because
   * the value may be defined in some cases where T is not actually a constant matrix.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct constant_coefficient
  {
    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};
  };


  /**
   * \brief Deduction guide for \ref constant_coefficient.
   */
  template<typename T>
  explicit constant_coefficient(const T&) -> constant_coefficient<T>;


  /**
   * \brief Helper template for constant_coefficient.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  constexpr auto constant_coefficient_v = constant_coefficient<T>::value;


  // ----------------- //
  //  constant_matrix  //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct scalar_status_is : std::false_type {};

    template<typename T, Likelihood b>
    struct scalar_status_is<T, b, std::enable_if_t<std::decay_t<T>::status == b>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that all elements of an object are the same constant value.
   */
  template<typename T, CompileTimeStatus c = CompileTimeStatus::any, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept constant_matrix = indexible<T> and scalar_constant<constant_coefficient<T>, c> and
    (b == Likelihood::maybe or constant_coefficient<T>::status == b);
#else
  constexpr bool constant_matrix = indexible<T> and scalar_constant<constant_coefficient<T>, c> and
    (b == Likelihood::maybe or detail::scalar_status_is<constant_coefficient<T>, b>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_MATRIX_HPP
