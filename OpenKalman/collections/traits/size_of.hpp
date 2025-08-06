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
 * \brief Definition for \ref collections::size_of.
 */

#ifndef OPENKALMAN_COLLECTIONS_SIZE_OF_HPP
#define OPENKALMAN_COLLECTIONS_SIZE_OF_HPP

#include "values/values.hpp"
#include "collections/concepts/sized.hpp"
#include "collections/functions/get_size.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief The size of a \ref sized object (including a \ref collection).
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct size_of {};


#ifdef __cpp_concepts
  template<sized T>
  struct size_of<T>
#else
  template<typename T>
  struct size_of<T, std::enable_if_t<sized<T> and not values::fixed<decltype(collections::get_size(std::declval<T>()))>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<sized T> requires values::fixed<decltype(collections::get_size(std::declval<T>()))>
  struct size_of<T>
#else
  template<typename T>
  struct size_of<T, std::enable_if_t<sized<T> and values::fixed<decltype(collections::get_size(std::declval<T>()))>>>
#endif
    : values::fixed_number_of<decltype(collections::get_size(std::declval<T>()))> {};


  /**
   * \brief Helper for \ref collections::size_of.
   */
  template<typename T>
  inline constexpr std::size_t size_of_v = size_of<T>::value;


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_SIZE_OF_HPP
