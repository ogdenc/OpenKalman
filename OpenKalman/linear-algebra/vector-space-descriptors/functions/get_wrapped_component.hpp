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
 * \brief Definition for \ref get_wrapped_component.
 */

#ifndef OPENKALMAN_WRAP_GET_ELEMENT_HPP
#define OPENKALMAN_WRAP_GET_ELEMENT_HPP

#include <type_traits>
#include "basics/values/values.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/static_vector_space_descriptor_traits.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/dynamic_vector_space_descriptor_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"


namespace OpenKalman
{
  /**
   * \brief Gets an element from a matrix or tensor object and wraps the result.
   * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
   * or in other words, performing <code>to_euclidean_element</code> followed by <code>from_euclidean_element<code>.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param local_index A local index accessing the element.
   * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
   */
#ifdef __cpp_concepts
  constexpr value::number auto
  get_wrapped_component(const vector_space_descriptor auto& t, const auto& g, std::size_t local_index, std::size_t start)
  requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
  template<typename T, typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
  constexpr auto
  get_wrapped_component(const T& t, const G& g, std::size_t local_index, std::size_t start)
#endif
  {
    using T_d = std::decay_t<decltype(t)>;
    if constexpr (euclidean_vector_space_descriptor<T_d>)
      return g(start + local_index);
    else if constexpr (static_vector_space_descriptor<T_d>)
      return interface::static_vector_space_descriptor_traits<T_d>::get_wrapped_component(g, local_index, start);
    else
      return interface::dynamic_vector_space_descriptor_traits<T_d>{t}.get_wrapped_component(g, local_index, start);
  }


} // namespace OpenKalman


#endif //OPENKALMAN_WRAP_GET_ELEMENT_HPP
