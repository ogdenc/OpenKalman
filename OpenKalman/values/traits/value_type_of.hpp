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
 * \brief Definition for \ref value:value_type_of and \ref value:value_type_of_t.
 */

#ifndef OPENKALMAN_VALUES_VALUE_TYPE_OF_HPP
#define OPENKALMAN_VALUES_VALUE_TYPE_OF_HPP

#include <type_traits>
#include "values/functions/to_value_type.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Obtain the underlying value type associated with a \ref value.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct value_type_of {};


  /// \overload
#ifdef __cpp_concepts
  template<typename T> requires requires(T t) { to_value_type(t); }
  struct value_type_of<T>
#else
  template<typename T>
  struct value_type_of<T, std::void_t<decltype(to_value_type(std::declval<T>()))>>
#endif
  {
    using type = std::decay_t<decltype(to_value_type(std::declval<T>()))>;
  };


  /**
   * \brief Helper template for \ref value_type_of.
   */
  template<typename T>
  using value_type_of_t = typename value_type_of<T>::type;

}

#endif
