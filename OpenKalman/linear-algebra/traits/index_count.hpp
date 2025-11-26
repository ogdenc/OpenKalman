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
 * \brief Definition for \ref index_count.
 */

#ifndef OPENKALMAN_INDEX_COUNT_HPP
#define OPENKALMAN_INDEX_COUNT_HPP

#include "count_indices.hpp"

namespace OpenKalman
{
  /**
   * \brief The minimum number of indices needed to access all the components of an object (i.e., the rank or order).
   * \details If T is \ref indexible, the index_count will be a fixed \ref values::index.
   * \internal \sa interface::object_traits::count_indices
   * \tparam T An \indexible object (tensor, vector, matrix, etc.)
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct index_count {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T> requires requires(const T& t) { {count_indices(t)} -> values::fixed; }
  struct index_count<T>
#else
  template<typename T>
  struct index_count<T, std::enable_if_t<values::fixed<decltype(count_indices(std::declval<const T&>()))>>>
#endif
    : std::integral_constant<std::size_t, values::fixed_value_of_v<decltype(count_indices(std::declval<const T&>()))>> {};


  /**
   * \brief helper template for \ref index_count.
   */
  template<typename T>
  static constexpr std::size_t index_count_v = index_count<T>::value;


}

#endif
