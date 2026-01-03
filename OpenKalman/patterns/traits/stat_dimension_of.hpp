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
 * \brief Definition for \ref patterns::stat_dimension_of.
 */

#ifndef OPENKALMAN_PATTERNS_STAT_DIMENSION_OF_HPP
#define OPENKALMAN_PATTERNS_STAT_DIMENSION_OF_HPP

#include <type_traits>
#include "values/concepts/fixed.hpp"
#include "values/functions/to_value_type.hpp"
#include "patterns/functions/get_stat_dimension.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief The dimension size of a set of \ref patterns::pattern if it is transformed into Euclidean space.
   * \details The associated static member <code>value</code> is the size of the \ref patterns::pattern when transformed
   * to Euclidean space, or \ref stdex::dynamic_extent if not known at compile time.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct stat_dimension_of {};


#ifdef __cpp_concepts
  template<pattern T> requires requires(T&& t) { {patterns::get_stat_dimension(t)} -> values::index; }
  struct stat_dimension_of<T>
#else
  template<typename T>
  struct stat_dimension_of<T, std::enable_if_t<values::index<decltype(patterns::get_stat_dimension(std::declval<T>()))>>>
#endif
  : std::conditional_t<
        values::fixed<decltype(patterns::get_stat_dimension(std::declval<T>()))>,
        values::fixed_value_of<decltype(patterns::get_stat_dimension(std::declval<T>()))>,
        std::integral_constant<std::size_t, stdex::dynamic_extent>> {};


  /**
   * \brief Helper template for \ref patterns::stat_dimension_of.
   */
  template<typename T>
  constexpr auto stat_dimension_of_v = stat_dimension_of<T>::value;


}

#endif
