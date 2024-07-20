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
 * \brief Definition for \ref index_dimension_of.
 */

#ifndef OPENKALMAN_INDEX_DIMENSION_OF_HPP
#define OPENKALMAN_INDEX_DIMENSION_OF_HPP


namespace OpenKalman
{
  /**
   * \brief The dimension of an index for a matrix, expression, or array.
   * \details The static constexpr <code>value</code> member indicates the size of the object associated with a
   * particular index. If the dimension is dynamic, <code>value</code> will be \ref dynamic_size.
   * \tparam N The index
   * \tparam T The matrix, expression, or array
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N = 0>
#else
  template<typename T, std::size_t N = 0, typename = void>
#endif
  struct index_dimension_of;


#ifdef __cpp_concepts
  template<typename T, std::size_t N> requires requires(T t) { {get_index_dimension_of<N>(t)} -> dynamic_index_value; }
  struct index_dimension_of<T, N>
#else
  template<typename T, std::size_t N>
  struct index_dimension_of<T, N, std::enable_if_t<indexible<T> and dynamic_index_value<decltype(get_index_dimension_of<N>(std::declval<T>()))>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<typename T, std::size_t N> requires requires(T t) { {get_index_dimension_of<N>(t)} -> static_index_value; }
  struct index_dimension_of<T, N>
#else
  template<typename T, std::size_t N>
  struct index_dimension_of<T, N, std::enable_if_t<indexible<T> and static_index_value<decltype(get_index_dimension_of<N>(std::declval<T>()))>>>
#endif
    : std::integral_constant<std::size_t, std::decay_t<decltype(get_index_dimension_of<N>(std::declval<T>()))>::value> {};


  /**
   * \brief helper template for \ref index_dimension_of.
   */
  template<typename T, std::size_t N = 0>
  static constexpr auto index_dimension_of_v = index_dimension_of<T, N>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DIMENSION_OF_HPP
