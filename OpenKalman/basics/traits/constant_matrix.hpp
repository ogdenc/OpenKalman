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

  /**
   * \brief Specifies that all components of an object are the same constant value.
   */
  template<typename T, ConstantType c = ConstantType::any>
#ifdef __cpp_concepts
  concept constant_matrix = indexible<T> and
#else
  constexpr bool constant_matrix =
#endif
    (scalar_constant<constant_coefficient<T>, c> or (c != ConstantType::static_constant and one_dimensional<T>));


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_MATRIX_HPP
