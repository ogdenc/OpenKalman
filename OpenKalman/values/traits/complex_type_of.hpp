/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value:complex_type_of.
 */

#ifndef OPENKALMAN_VALUES_COMPLEX_TYPE_OF_T_HPP
#define OPENKALMAN_VALUES_COMPLEX_TYPE_OF_T_HPP

#include <type_traits>
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/functions/internal/make_complex_number.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Obtain the \ref values::complex "complex" type associated with a \ref values::value "value", if it exists.
   * \details This type will be equivalent to
   * \code
   * std::decay_t<decltype(values::internal::make_complex_number<number_type_of_t<T>>(std::declval<T>()))>
   * /endcode.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct complex_type_of {};


  /// \overload
#ifdef __cpp_concepts
  template<value T> requires requires { values::internal::make_complex_number<number_type_of_t<T>>(std::declval<T>()); }
  struct complex_type_of<T>
#else
  template<typename T>
  struct complex_type_of<T, std::void_t<decltype(values::internal::make_complex_number<number_type_of_t<T>>(std::declval<T>()))>>
#endif
  {
    using type = std::decay_t<decltype(values::internal::make_complex_number<number_type_of_t<T>>(std::declval<T>()))>;
  };


  /**
   * \brief Helper template for \ref complex_type_of.
   */
  template<typename T>
  using complex_type_of_t = typename complex_type_of<T>::type;

}

#endif
