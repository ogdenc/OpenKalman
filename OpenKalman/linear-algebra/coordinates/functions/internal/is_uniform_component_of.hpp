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


namespace OpenKalman::coordinate::internal
{
  /**
   * \internal
   * \brief Whether <code>a</code> is a 1D \ref coordinate::pattern object that, when replicated some number of times, becomes <code>c</code>.
   */
#ifdef __cpp_concepts
  template<coordinate::pattern A, coordinate::pattern B>
#else
  template<typename A, typename B, std::enable_if_t<coordinate::pattern<A> and coordinate::pattern<B>, int> = 0>
#endif
  constexpr bool is_uniform_component_of(const A& a, const B& b)
  {
    if constexpr (fixed_pattern<A> and fixed_pattern<B>)
      return equivalent_to_uniform_static_vector_space_descriptor_component_of<A, B>;
    else if constexpr (euclidean_pattern<A> and euclidean_pattern<B>)
      return get_size(a) == 1;
    else
      return false;
  }


  /**
   * \internal
   * \overload
   */
  template<typename S1, typename S2>
  constexpr bool is_uniform_component_of(const coordinate::DynamicDescriptor<S1>& a, const coordinate::DynamicDescriptor<S2>& b)
  {
    if constexpr (not std::is_same_v<S1, S2>) return false;
    else if (get_size(a) != 1) return false;
    else if (get_is_euclidean(a) and get_is_euclidean(b)) return true;
    else return a * coordinate::get_size(b) == b;
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template<coordinate::pattern A, typename S>
#else
  template<typename A, typename S, std::enable_if_t<coordinate::pattern<A>, int> = 0>
#endif
  constexpr bool is_uniform_component_of(const A& a, const coordinate::DynamicDescriptor<S>& b)
  {
    if (get_size(a) != 1) return false;
    else if (get_is_euclidean(a) and get_is_euclidean(b)) return true;
    else return a * coordinate::get_size(b) == b;
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template<typename S, coordinate::pattern B>
#else
  template<typename S, typename B, std::enable_if_t<coordinate::pattern<B>, int> = 0>
#endif
  constexpr bool is_uniform_component_of(const coordinate::DynamicDescriptor<S>& a, const B& b)
  {
    if (get_size(a) != 1) return false;
    else if (get_is_euclidean(a) and get_is_euclidean(b)) return true;
    else return a * coordinate::get_size(b) == b;
  }


} // namespace OpenKalman::coordinate::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_FUNCTIONS_HPP
