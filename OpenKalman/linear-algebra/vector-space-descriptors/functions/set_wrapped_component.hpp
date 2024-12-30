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
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief Set a component and then perform any required wrapping.
   * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
   * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param x The new value to be set.
   * \param local_index A local index accessing the element.
   * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, value::value X, value::index L, value::index S>
  constexpr void
  set_wrapped_component(const T& t, const auto& s, const auto& g, const X& x, const L& local_index, const S& start)
  requires requires { s(x, start); s(g(start), start); }
#else
  template<typename T, typename Setter, typename Getter, typename X, typename L, typename S, std::enable_if_t<
    vector_space_descriptor<T> and value::value<X> and value::index<L> and value::index<S> and
    std::is_invocable<const Setter&, const X&, const S&>::value and
    std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, const S&>::type, const S&>::value, int> = 0>
  constexpr void
  set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index, const S& start)
#endif
  {
    if constexpr (static_vector_space_descriptor<T> and value::fixed<L>)
      static_assert(value::to_number(local_index) < dimension_size_of_v<T>);

    if constexpr (euclidean_vector_space_descriptor<T>) s(x, start + local_index);
    else interface::vector_space_traits<T>::set_wrapped_component(t, s, g, x, local_index, start);
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_SET_WRAPPED_COMPONENT_HPP
