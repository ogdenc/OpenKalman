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
 * \brief Definition for \ref to_stat_space.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP
#define OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP

#include <type_traits>
#include <functional>
#include "values/concepts/value.hpp"
#include "values/classes/operation.hpp"
#include "collections/functions/get.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/internal/get_euclidean_index_table.hpp"
#include "linear-algebra/coordinates/functions/internal/get_component_start_indices.hpp"
#include "linear-algebra/coordinates/functions/internal/get_euclidean_component_start_indices.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_collection_element.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Maps an element from coordinates in modular space to coordinates in Euclidean space.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref values::number
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
   */
#ifdef __cpp_concepts
  template<pattern T, collections::collection R>
  constexpr collections::collection decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<pattern<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  to_stat_space(const T& t, R&& data_range)
  {
    if constexpr (stat_dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(stat_dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (euclidean_pattern<T>)
    {
      return std::forward<R>(data_range);
    }
    else if constexpr (descriptor<T>)
    {
#if __cplusplus >= 202002L
      return std::invoke(interface::coordinate_descriptor_traits<T>::to_stat_space, t, std::forward<R>(data_range));
#else
      return interface::coordinate_descriptor_traits<T>::to_stat_space(t, std::forward<R>(data_range));;
#endif
    }
    else // if constexpr (descriptor_collection<T>)
    {
      return collections::views::concat(std::forward<R>(data_range));

      // Can we just concatenate the to_stat_space(...) result for each descriptor in the collection?

      auto component_ix = collections::get(internal::get_euclidean_index_table(t), euclidean_local_index);
      auto component = internal::get_descriptor_collection_element(t, component_ix);
      auto start_i = collections::get(internal::get_component_start_indices(t), component_ix);
      auto start_e = collections::get(internal::get_euclidean_component_start_indices(t), component_ix);
      auto new_g = [&g, start_i](auto i) { return g(values::operation {std::plus{}, start_i, i}); };
      auto new_euclidean_local_index = values::operation {std::minus{}, euclidean_local_index, start_e};
      return to_stat_space(component, new_g, new_euclidean_local_index);
    }
  }


} // namespace OpenKalman::coordinates


#endif //OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP
