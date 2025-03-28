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
 * \brief Definition for \ref coordinate::euclidean_size_of.
 */

#ifndef OPENKALMAN_COORDINATE_EUCLIDEAN_SIZE_OF_HPP
#define OPENKALMAN_COORDINATE_EUCLIDEAN_SIZE_OF_HPP

#include <type_traits>
#include "values/concepts/fixed.hpp"
#include "values/functions/to_number.hpp"
#include "linear-algebra/coordinates/functions/get_euclidean_size.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief The dimension size of a set of \ref coordinate::pattern if it is transformed into Euclidean space.
   * \details The associated static member <code>value</code> is the size of the \ref coordinate::pattern when transformed
   * to Euclidean space, or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<pattern T>
  struct euclidean_size_of : std::integral_constant<std::size_t, dynamic_size> {};
#else
  template<typename T, typename = void>
  struct euclidean_size_of {};
#endif


#ifndef __cpp_concepts
  template<typename T>
  struct euclidean_size_of<T, std::enable_if_t<not value::fixed<decltype(get_euclidean_size(std::declval<std::decay_t<T>>()))>>>
    : std::integral_constant<std::size_t, dynamic_size> {};
#endif


#ifdef __cpp_concepts
  template<pattern T> requires value::fixed<decltype(get_euclidean_size(std::declval<std::decay_t<T>>()))>
  struct euclidean_size_of<T>
#else
  template<typename T>
  struct euclidean_size_of<T, std::enable_if_t<value::fixed<decltype(get_euclidean_size(std::declval<std::decay_t<T>>()))>>>
#endif
    : std::integral_constant<std::size_t, value::to_number(get_euclidean_size(T{}))> {};


  /**
   * \brief Helper template for \ref coordinate::euclidean_size_of.
   */
  template<typename T>
  constexpr auto euclidean_size_of_v = euclidean_size_of<T>::value;


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_COORDINATE_EUCLIDEAN_SIZE_OF_HPP
