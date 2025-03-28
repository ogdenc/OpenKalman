/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinate::component_count_of.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP
#define OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP

#include <type_traits>
#include "values/concepts/fixed.hpp"
#include "linear-algebra/coordinates/functions/get_component_count.hpp"


namespace OpenKalman::coordinate
{
  /**
   * \brief The number of atomic \ref coordinate::descriptor "descriptors" of a \ref coordinate::pattern.
   * \details The result is \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<pattern T>
  struct component_count_of : std::integral_constant<std::size_t, dynamic_size> {};
#else
  template<typename T, typename = void>
  struct component_count_of {};
#endif


#ifndef __cpp_concepts
  template<typename T>
  struct component_count_of<T, std::enable_if_t<not value::fixed<decltype(get_component_count(std::declval<std::decay_t<T>>()))>>>
    : std::integral_constant<std::size_t, dynamic_size> {};
#endif


#ifdef __cpp_concepts
  template<pattern T> requires value::fixed<decltype(get_component_count(std::declval<std::decay_t<T>>()))>
  struct component_count_of<T>
#else
  template<typename T>
  struct component_count_of<T, std::enable_if_t<value::fixed<decltype(get_component_count(std::declval<std::decay_t<T>>()))>>>
#endif
    : std::integral_constant<std::size_t, value::fixed_number_of_v<decltype(get_component_count(std::declval<std::decay_t<T>>()))>> {};


  /**
   * \brief Helper template for \ref coordinate::component_count_of.
   */
  template<typename T>
  constexpr auto component_count_of_v = component_count_of<T>::value;


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP
