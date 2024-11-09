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
 * \brief Definition for \ref set_wrapped_component.
 */

#ifndef OPENKALMAN_SET_WRAPPED_COMPONENT_HPP
#define OPENKALMAN_SET_WRAPPED_COMPONENT_HPP

#include <type_traits>


namespace OpenKalman
{
  /**
   * \brief Set a component and then perform any required wrapping.
   * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
   * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param x The new value to be set.
   * \param local_index A local index accessing the element.
   * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
   */
#ifdef __cpp_concepts
  constexpr void set_wrapped_component(const vector_space_descriptor auto& t, const auto& s, const auto& g,
    const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
  requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
  template<typename T, typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
    std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
  constexpr void set_wrapped_component(const T& t, const S& s, const G& g,
    const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x, std::size_t local_index, std::size_t start)
#endif
  {
    using T_d = std::decay_t<decltype(t)>;
    if constexpr (euclidean_vector_space_descriptor<T_d>)
      s(x, start + local_index);
    else if constexpr (static_vector_space_descriptor<T_d>)
      static_vector_space_descriptor_traits<T_d>::set_wrapped_component(s, g, x, local_index, start);
    else
      dynamic_vector_space_descriptor_traits<T_d>{t}.set_wrapped_component(s, g, x, local_index, start);
  }


} // namespace OpenKalman


#endif //OPENKALMAN_SET_WRAPPED_COMPONENT_HPP
