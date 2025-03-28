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
 * \brief Definition for \ref to_euclidean_element.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP
#define OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP

#include <type_traits>
#include <functional>
#include "values/concepts/value.hpp"
#include "values/functions/internal/get_collection_element.hpp"
#include "values/classes/operation.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/internal/get_euclidean_index_table.hpp"
#include "linear-algebra/coordinates/functions/internal/get_component_start_indices.hpp"
#include "linear-algebra/coordinates/functions/internal/get_euclidean_component_start_indices.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_collection_element.hpp"
#include "linear-algebra/coordinates/traits/euclidean_size_of.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief Maps an element from coordinates in modular space to coordinates in Euclidean space.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
   */
#ifdef __cpp_concepts
  template<pattern T, value::index L>
  constexpr value::value auto
  to_euclidean_element(const T& t, const auto& g, const L& euclidean_local_index)
  requires requires(std::size_t i) { {g(i)} -> value::value; }
#else
  template<typename T, typename Getter, typename L, std::enable_if_t<pattern<T> and value::index<L> and
    value::value<typename std::invoke_result<Getter, std::size_t>::type>, int> = 0>
  constexpr auto
  to_euclidean_element(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
  {
    if constexpr (euclidean_size_of_v<T> != dynamic_size and value::fixed<L>)
      static_assert(value::to_number(euclidean_local_index) < euclidean_size_of_v<T>);

    if constexpr (euclidean_pattern<T>)
    {
      return g(euclidean_local_index);
    }
    else if constexpr (descriptor<T>)
    {
      return interface::coordinate_descriptor_traits<T>::to_euclidean_component(t, g, euclidean_local_index);;
    }
    else // if constexpr (descriptor_collection<T>)
    {
      auto component_ix = value::internal::get_collection_element(internal::get_euclidean_index_table(t), euclidean_local_index);
      auto component = internal::get_descriptor_collection_element(t, component_ix);
      auto start_i = value::internal::get_collection_element(internal::get_component_start_indices(t), component_ix);
      auto start_e = value::internal::get_collection_element(internal::get_euclidean_component_start_indices(t), component_ix);
      auto new_g = [&g, start_i](auto i) { return g(value::operation {std::plus{}, start_i, i}); };
      auto new_euclidean_local_index = value::operation {std::minus{}, euclidean_local_index, start_e};
      return to_euclidean_element(component, new_g, new_euclidean_local_index);
    }
  }


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP
