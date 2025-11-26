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
 * \brief Definition for \ref constant_value_of.
 */

#ifndef OPENKALMAN_CONSTANT_VALUE_OF_HPP
#define OPENKALMAN_CONSTANT_VALUE_OF_HPP

#include "linear-algebra/traits/constant_value.hpp"

namespace OpenKalman
{
  /**
   * \brief The static \ref constant value of an indexible object, if it exists.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct constant_value_of {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<typename T> requires values::fixed<decltype(constant_value(std::declval<T>()))>
  struct constant_value_of<T>
#else
  template<typename T>
  struct constant_value_of<T, std::enable_if_t<values::fixed<decltype(constant_value(std::declval<T>()))>>>
#endif
    : values::fixed_value_of<decltype(constant_value(std::declval<T>()))> {};


  /**
   * \brief helper template for \ref constant_value_of.
   */
  template<typename T>
  constexpr auto constant_value_of_v = constant_value_of<T>::value;

}

#endif
