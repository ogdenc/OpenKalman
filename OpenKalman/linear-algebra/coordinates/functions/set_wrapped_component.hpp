/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
#include <functional>
#include "values/concepts/value.hpp"
#include "values/classes/operation.hpp"
#include "collections/functions/get.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/internal/get_index_table.hpp"
#include "linear-algebra/coordinates/functions/internal/get_component_start_indices.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_collection_element.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Set a component and then perform any required wrapping.
   * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
   * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref values::number
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param x The new value to be set.
   * \param local_index A local index accessing the element.
   */
#ifdef __cpp_concepts
  template<pattern T, values::value X, values::index L>
  constexpr void
  set_wrapped_component(const T& t, const auto& s, const auto& g, const X& x, const L& local_index)
  requires requires(std::size_t i) { s(x, i); s(g(i), i); }
#else
  template<typename T, typename Setter, typename Getter, typename X, typename L, std::enable_if_t<
    pattern<T> and values::value<X> and values::index<L> and
    std::is_invocable<const Setter&, const X&, std::size_t>::value and
    std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
  constexpr void
  set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index)
#endif
  {
    if constexpr (dimension_of_v<T> != dynamic_size and values::fixed<L>)
      static_assert(values::to_number(local_index) < dimension_of_v<T>);

    if constexpr (euclidean_pattern<T>)
    {
      s(x, local_index);
    }
    else if constexpr (descriptor<T>)
    {
      interface::coordinate_descriptor_traits<T>::set_wrapped_component(t, s, g, x, local_index);;
    }
    else // if constexpr (descriptor_collection<T>)
    {
      using Scalar = std::decay_t<decltype(x)>;
      auto component_ix = collections::get(internal::get_index_table(t), local_index);
      auto component = internal::get_descriptor_collection_element(t, component_ix);
      auto start_i = collections::get(internal::get_component_start_indices(t), component_ix);
      auto new_g = [&g, start_i](auto i) { return g(values::operation {std::plus{}, start_i, i}); };
      auto new_s = [&s, start_i](const Scalar& x, auto i) { s(x, values::operation {std::plus{}, start_i, i}); };
      auto new_local_index = values::operation {std::minus{}, local_index, start_i};
      set_wrapped_component(component, new_s, new_g, x, new_local_index);
    }
  }


} // namespace OpenKalman::coordinates


#endif //OPENKALMAN_SET_WRAPPED_COMPONENT_HPP
