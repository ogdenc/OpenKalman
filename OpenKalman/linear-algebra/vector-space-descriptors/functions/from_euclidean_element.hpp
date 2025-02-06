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
 * \brief Definition for \ref from_euclidean_element.
 */

#ifndef OPENKALMAN_FROM_EUCLIDEAN_ELEMENT_HPP
#define OPENKALMAN_FROM_EUCLIDEAN_ELEMENT_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief The inverse of <code>to_euclidean_element</code>. Maps coordinates in Euclidean space back into modular space.
   * \param g An element getter mapping an index i of type std::size_t to an element value
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param local_index A local index accessing the coordinate in modular space.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, value::index L>
  constexpr value::value auto
  from_euclidean_element(const T& t, const auto& g, const L& local_index)
  requires requires(std::size_t i) { {g(i)} -> value::value; }
#else
  template<typename T, typename Getter, typename L, std::enable_if_t<vector_space_descriptor<T> and value::index<L> and
    value::value<typename std::invoke_result<Getter, std::size_t>::type>, int> = 0>
  constexpr auto
  from_euclidean_element(const T& t, const Getter& g, const L& local_index)
#endif
  {
    if constexpr (static_vector_space_descriptor<T> and value::fixed<L>)
      static_assert(value::to_number(local_index) < dimension_size_of_v<T>);

    if constexpr (euclidean_vector_space_descriptor<T>) return g(local_index);
    else return interface::vector_space_traits<T>::from_euclidean_component(t, g, local_index);
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_FROM_EUCLIDEAN_ELEMENT_HPP
