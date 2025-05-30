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
 * \brief Definition for \ref coordinates::dimension_of.
 */

#ifndef OPENKALMAN_COORDINATE_SIZE_OF_HPP
#define OPENKALMAN_COORDINATE_SIZE_OF_HPP

#include <type_traits>
#include "basics/global-definitions.hpp"
#include "values/concepts/fixed.hpp"
#include "values/functions/to_number.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief The size of a \ref coordinates::pattern.
   * \details The associated static member <code>value</code> is the size of the \ref coordinates::pattern,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<pattern T>
  struct dimension_of : std::integral_constant<std::size_t, dynamic_size> {};
#else
  template<typename T, typename = void>
  struct dimension_of {};
#endif


#ifndef __cpp_concepts
  template<typename T>
  struct dimension_of<T, std::enable_if_t<not values::fixed<decltype(get_dimension(std::declval<std::decay_t<T>>()))>>>
    : std::integral_constant<std::size_t, dynamic_size> {};
#endif


#ifdef __cpp_concepts
  template<pattern T> requires values::fixed<decltype(get_dimension(std::declval<std::decay_t<T>>()))>
  struct dimension_of<T>
#else
  template<typename T>
  struct dimension_of<T, std::enable_if_t<values::fixed<decltype(get_dimension(std::declval<std::decay_t<T>>()))>>>
#endif
    : std::integral_constant<std::size_t, values::to_number(get_dimension(std::decay_t<T>{}))> {};


  /**
   * \brief Helper template for \ref coordinates::dimension_of.
   */
  template<typename T>
  constexpr auto dimension_of_v = dimension_of<T>::value;


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_COORDINATE_SIZE_OF_HPP
