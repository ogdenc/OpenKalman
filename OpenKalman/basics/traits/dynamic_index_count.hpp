/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref dynamic_index_count.
 */

#ifndef OPENKALMAN_DYNAMIC_INDEX_COUNT_HPP
#define OPENKALMAN_DYNAMIC_INDEX_COUNT_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr std::size_t dynamic_index_count_impl(std::index_sequence<I...>)
    {
      return ((dynamic_dimension<T, I> ? 1 : 0) + ... + 0);
    }
  }


  /**
   * \brief Counts the number of indices of T in which the dimensions are dynamic.
   * \details If \ref index_count_v is itself dynamic, the result is \ref dynamic_size.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct dynamic_index_count : std::integral_constant<std::size_t, dynamic_size> {};


  /**
   * \overload
   * \brief Case in which the number of indices is static.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (index_count_v<T> != dynamic_size)
  struct dynamic_index_count<T>
#else
  template<typename T>
  struct dynamic_index_count<T, std::enable_if_t<(index_count<T>::value != dynamic_size)>>
#endif
    : std::integral_constant<std::size_t, detail::dynamic_index_count_impl<T>(std::make_index_sequence<index_count_v<T>> {})> {};


  /**
   * \brief Helper template for \ref dynamic_index_count
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  static constexpr std::size_t dynamic_index_count_v = dynamic_index_count<T>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_DYNAMIC_INDEX_COUNT_HPP
