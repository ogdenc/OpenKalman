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
 * \brief Definition for \ref vector_space_descriptor_of.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_OF_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_OF_HPP


namespace OpenKalman
{
  /**
   * \brief The \ref vector_space_descriptor for index N of object T.
   * \details Usually, this is defined by the traits for T.
   * \tparam T A matrix, expression, or array
   * \tparam N An index number (0 = rows, 1 = columns, etc.)
   * \internal \sa interface::indexible_object_traits::get_vector_space_descriptor
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N = 0>
#else
  template<typename T, std::size_t N = 0, typename = void>
#endif
  struct vector_space_descriptor_of {};


  /**
   * \overload As defined by the traits for T.
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t N> requires requires(T t) { {get_vector_space_descriptor<N>(t)} -> vector_space_descriptor; }
  struct vector_space_descriptor_of<T, N>
#else
  template<typename T, std::size_t N>
  struct vector_space_descriptor_of<T, N, std::enable_if_t<vector_space_descriptor<decltype(get_vector_space_descriptor<N>(std::declval<T>()))>>>
#endif
  {
    using type = std::decay_t<decltype(get_vector_space_descriptor<N>(std::declval<T>()))>;
  };


  /**
   * \brief helper template for \ref vector_space_descriptor_of.
   */
  template<typename T, std::size_t N>
  using vector_space_descriptor_of_t = typename vector_space_descriptor_of<T, N>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_OF_HPP
