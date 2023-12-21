/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Functions for \ref is_uniform_component_of objects.
 */

#ifndef OPENKALMAN_IS_UNIFORM_COMPONENT_OF_HPP
#define OPENKALMAN_IS_UNIFORM_COMPONENT_OF_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Whether <code>a</code> is a 1D \ref vector_space_descriptor object that, when replicated some number of times, becomes <code>c</code>.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor A, vector_space_descriptor C>
#else
  template<typename A, typename C, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<C>, int> = 0>
#endif
  constexpr bool is_uniform_component_of(const A& a, const C& c)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<C>)
      return equivalent_to_uniform_dimension_type_of<A, C>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<C>)
      return get_dimension_size_of(a) == 1;
    else
      return false;
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor A, typename...S>
#else
  template<typename A, typename...S, std::enable_if_t<vector_space_descriptor<A>, int> = 0>
#endif
  constexpr bool is_uniform_component_of(const A& a, const DynamicTypedIndex<S...>& c)
  {
    if (get_dimension_size_of(a) != 1) return false;
    else if (get_vector_space_descriptor_is_euclidean(a) and get_vector_space_descriptor_is_euclidean(c)) return true;
    else return replicate_vector_space_descriptor<S...>(a, get_dimension_size_of(c)) == c;
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template<typename...T, vector_space_descriptor C>
#else
  template<typename...T, typename C, std::enable_if_t<vector_space_descriptor<C>, int> = 0>
#endif
  constexpr bool is_uniform_component_of(const DynamicTypedIndex<T...>& a, const C& c)
  {
    if (get_dimension_size_of(a) != 1) return false;
    else if (get_vector_space_descriptor_is_euclidean(a) and get_vector_space_descriptor_is_euclidean(c)) return true;
    else return replicate_vector_space_descriptor(a, get_dimension_size_of(c)) == c;
  }


  /**
   * \internal
   * \overload
   */
  template<typename...T, typename...S>
  constexpr bool is_uniform_component_of(const DynamicTypedIndex<T...>& a, const DynamicTypedIndex<S...>& c)
  {
    if constexpr (((not std::is_same_v<T, S>) or ...)) return false;
    else if (get_dimension_size_of(a) != 1) return false;
    else if (get_vector_space_descriptor_is_euclidean(a) and get_vector_space_descriptor_is_euclidean(c)) return true;
    else return replicate_vector_space_descriptor(a, get_dimension_size_of(c)) == c;
  }


} // namespace OPENKALMAN_IS_UNIFORM_COMPONENT_OF_HPP::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_FUNCTIONS_HPP
