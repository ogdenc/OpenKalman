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

#ifndef OPENKALMAN_DIMENSION_OF_HPP
#define OPENKALMAN_DIMENSION_OF_HPP

#include <type_traits>
#include "values/values.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief The size of a \ref coordinates::pattern.
   * \details The associated static member <code>value</code> is the size of the \ref coordinates::pattern,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_of {};


#ifdef __cpp_concepts
  template<pattern T> requires requires(T t) { {coordinates::get_dimension(t)} -> values::index; }
  struct dimension_of<T>
#else
  template<typename T>
  struct dimension_of<T, std::enable_if_t<values::index<decltype(coordinates::get_dimension(std::declval<T>()))>>>
#endif
    : std::conditional_t<
        values::fixed<decltype(coordinates::get_dimension(std::declval<T>()))>,
        values::fixed_number_of<decltype(coordinates::get_dimension(std::declval<T>()))>,
        std::integral_constant<std::size_t, dynamic_size>> {};


  /**
   * \brief Helper template for \ref coordinates::dimension_of.
   */
  template<typename T>
  constexpr auto dimension_of_v = dimension_of<T>::value;


}

#endif
