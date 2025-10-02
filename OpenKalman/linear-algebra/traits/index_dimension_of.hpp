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

#include "linear-algebra/traits/get_pattern_collection.hpp"

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
  struct index_dimension_of {};


#ifdef __cpp_concepts
  template<indexible T, std::size_t N>
  struct index_dimension_of<T, N>
#else
  template<typename T, std::size_t N>
  struct index_dimension_of<T, N, std::enable_if_t<indexible<T>>>
#endif
    : std::integral_constant<std::size_t, coordinates::dimension_of_v<decltype(get_pattern_collection<N>(std::declval<T>()))>> {};


  /**
   * \brief helper template for \ref index_dimension_of.
   */
  template<typename T, std::size_t N = 0>
  static constexpr auto index_dimension_of_v = index_dimension_of<T, N>::value;


}

#endif
