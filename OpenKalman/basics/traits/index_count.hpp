/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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


namespace OpenKalman
{
  /**
   * \brief The maximum number of indices of structure T. If dynamic, the result is OpenKalman::dynamic_size.
   * \internal \sa interface::indexible_object_traits::count_indices
   * \tparam T A tensor (vector, matrix, etc.)
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct index_count : std::integral_constant<std::size_t, 0> {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T> requires requires(T t) { {count_indices(t)} -> static_index_value; }
  struct index_count<T>
#else
  template<typename T>
  struct index_count<T, std::enable_if_t<indexible<T> and static_index_value<decltype(count_indices(std::declval<T>()))>>>
#endif
    : std::integral_constant<std::size_t, std::decay_t<decltype(count_indices(std::declval<T>()))>::value> {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T> requires requires(T t) { {count_indices(t)} -> dynamic_index_value; }
  struct index_count<T>
#else
  template<typename T>
  struct index_count<T, std::enable_if_t<indexible<T> and dynamic_index_value<decltype(count_indices(std::declval<T>()))>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size> {};


  /**
   * \brief helper template for \ref index_count.
   */
  template<typename T>
  static constexpr std::size_t index_count_v = index_count<T>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_COUNT_HPP
