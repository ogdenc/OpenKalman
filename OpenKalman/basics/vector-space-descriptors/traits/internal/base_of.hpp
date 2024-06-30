/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref base_of.
 */

#ifndef OPENKALMAN_BASE_OF_HPP
#define OPENKALMAN_BASE_OF_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief If T is a \ref internal::prefix_of "prefix of" or \ref internal::suffix_of "suffix of" U, return the base.
   */
#ifdef __cpp_concepts
  template<typename T, typename U>
#else
  template<typename T, typename U, typename = void>
#endif
  struct base_of {};


  /**
   * \brief Helper template for \ref base_of.
   */
  template<typename T, typename U>
  using base_of_t = typename base_of<T, U>::type;


#ifdef __cpp_concepts
  template<typename T, typename U> requires prefix_of<T, U>
  struct base_of<T, U>
#else
  template<typename T, typename U>
  struct base_of<T, U, std::enable_if_t<prefix_of<T, U>>>
#endif
  {
    using type = typename internal::is_prefix<T, U>::base;
  };


#ifdef __cpp_concepts
  template<typename T, typename U> requires suffix_of<T, U> and (not prefix_of<T, U>)
  struct base_of<T, U>
#else
  template<typename T, typename U>
  struct base_of<T, U, std::enable_if_t<(suffix_of<T, U> and not prefix_of<T, U>)>>
#endif
  {
    using type = reverse_fixed_vector_space_descriptor_t<base_of_t<reverse_fixed_vector_space_descriptor_t<T>, reverse_fixed_vector_space_descriptor_t<U>>>;
  };


} // namespace OpenKalman::internal

#endif //OPENKALMAN_BASE_OF_HPP
