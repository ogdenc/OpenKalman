/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_index_dimension_of function.
 */

#ifndef OPENKALMAN_GET_INDEX_DIMENSION_OF_HPP
#define OPENKALMAN_GET_INDEX_DIMENSION_OF_HPP


namespace OpenKalman
{
  /**
   * \brief Get the runtime dimensions of index N of \ref indexible T
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T, index_value N = std::integral_constant<std::size_t, 0>>
  constexpr index_value auto
#else
  template<typename T, typename N = std::integral_constant<std::size_t, 0>, std::enable_if_t<
    interface::count_indices_defined_for<T> and index_value<N>, int> = 0>
  constexpr auto
#endif
  get_index_dimension_of(const T& t, N n = N{})
  {
    return get_dimension_size_of(get_vector_space_descriptor(t, n));
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<std::size_t N, interface::count_indices_defined_for T>
  constexpr index_value auto
#else
  template<std::size_t N, typename T, std::enable_if_t<interface::count_indices_defined_for<T>, int> = 0>
  constexpr auto
#endif
  get_index_dimension_of(const T& t)
  {
    return get_dimension_size_of(get_vector_space_descriptor<N>(t));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_GET_INDEX_DIMENSION_OF_HPP
