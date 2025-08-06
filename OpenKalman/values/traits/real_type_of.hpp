/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value:real_type_of and \ref value:real_type_of_t.
 */

#ifndef OPENKALMAN_VALUES_REAL_TYPE_OF_HPP
#define OPENKALMAN_VALUES_REAL_TYPE_OF_HPP

#include <type_traits>
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/math/real.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Obtain the real type associated with a number (typically a \ref values::complex number.
   * \details This will be the type of the result of <code>values::to_number(values::real(std::declval<T>()))</code>.
   * Because some compilers allow integral complex numbers, you can apply this trait twice to ensure that the
   * result is a floating type. In that case, the first application will be integral and the second application
   * will be <code>double</code>.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct real_type_of {};


  /// \overload
#ifdef __cpp_concepts
  template<value T>
  struct real_type_of<T>
#else
  template<typename T>
  struct real_type_of<T, std::enable_if_t<value<T>>>
#endif
    : number_type_of<decltype(values::real(std::declval<T>()))> {};


  /**
   * \brief Helper template for \ref real_type_of.
   */
  template<typename T>
  using real_type_of_t = typename real_type_of<T>::type;


}

#endif
