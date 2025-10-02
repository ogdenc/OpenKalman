/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_index_extent function.
 */

#ifndef OPENKALMAN_GET_INDEX_EXTENT_HPP
#define OPENKALMAN_GET_INDEX_EXTENT_HPP

#include "linear-algebra/traits/get_index_pattern.hpp"

namespace OpenKalman
{
  /**
   * \brief Get the runtime dimensions of index N of \ref indexible T
   */
#ifdef __cpp_concepts
  template<indexible T, values::index N = std::integral_constant<std::size_t, 0>>
  constexpr values::index auto
#else
  template<typename T, typename N = std::integral_constant<std::size_t, 0>,
    std::enable_if_t<indexible<T> and values::index<N>, int> = 0>
  constexpr auto
#endif
  get_index_extent(T&& t, N n = N{})
  {
    return get_dimension(get_index_pattern(t, n));
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<std::size_t N, indexible T>
  constexpr values::index auto
#else
  template<std::size_t N, typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  get_index_extent(T&& t)
  {
    return get_dimension(get_index_pattern<N>(t));
  }


}

#endif
