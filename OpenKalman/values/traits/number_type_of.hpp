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
 * \brief Definition for \ref value:number_type_of and \ref value:number_type_of_t.
 */

#ifndef OPENKALMAN_VALUES_NUMBER_TYPE_OF_HPP
#define OPENKALMAN_VALUES_NUMBER_TYPE_OF_HPP

#include <type_traits>
#include "values/concepts/value.hpp"
#include "values/functions/to_number.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Obtain the \ref values::number type associated with a\ref values::value.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct number_type_of {};


  /// \overload
#ifdef __cpp_concepts
  template<value T>
  struct number_type_of<T>
#else
  template<typename T>
  struct number_type_of<T, std::enable_if_t<value<T>>>
#endif
  {
    using type = std::decay_t<decltype(to_number(std::declval<T>()))>;
  };


  /**
   * \brief Helper template for \ref number_type_of.
   */
  template<typename T>
  using number_type_of_t = typename number_type_of<T>::type;

}

#endif
