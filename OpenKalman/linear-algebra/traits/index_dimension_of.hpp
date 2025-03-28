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
  struct index_dimension_of {};


  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  requires requires { typename vector_space_descriptor_of_t<T, N>; }
  struct index_dimension_of<T, N>
#else
  struct index_dimension_of<T, N, std::void_t<typename vector_space_descriptor_of<T, N>::type>>
#endif
    : std::integral_constant<std::size_t, coordinate::size_of_v<vector_space_descriptor_of_t<T, N>>> {};


  /**
   * \brief helper template for \ref index_dimension_of.
   */
  template<typename T, std::size_t N = 0>
  static constexpr auto index_dimension_of_v = index_dimension_of<T, N>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DIMENSION_OF_HPP
