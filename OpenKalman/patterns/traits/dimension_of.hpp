/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::dimension_of.
 */

#ifndef OPENKALMAN_DIMENSION_OF_HPP
#define OPENKALMAN_DIMENSION_OF_HPP

#include <type_traits>
#include "values/values.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/functions/get_dimension.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief The size of a \ref patterns::pattern.
   * \details If T has a dimension, static member <code>value</code> holds that dimension.
   * If a dimension exists but it is not known at compile time, <code>value</code> is \ref stdex::dynamic_extent.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_of {};


#ifdef __cpp_concepts
  template<pattern T> requires requires(T&& t) { get_dimension(t); }
  struct dimension_of<T>
#else
  template<typename T>
  struct dimension_of<T, std::void_t<decltype(get_dimension(std::declval<T>()))>>
#endif
    : std::conditional_t<
        values::fixed<decltype(get_dimension(std::declval<T>()))>,
        values::fixed_value_of<decltype(get_dimension(std::declval<T>()))>,
        std::integral_constant<std::size_t, stdex::dynamic_extent>> {};


  /**
   * \brief Helper template for \ref patterns::dimension_of.
   */
  template<typename T>
  constexpr auto dimension_of_v = dimension_of<T>::value;


}

#endif
