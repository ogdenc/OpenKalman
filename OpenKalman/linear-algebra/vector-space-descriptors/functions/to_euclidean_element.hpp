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
 * \brief Definition for \ref to_euclidean_element.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP
#define OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief Maps an element from coordinates in modular space to coordinates in Euclidean space.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
   * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, value::index L, value::index S>
  constexpr value::value auto
  to_euclidean_element(const T& t, const auto& g, const L& euclidean_local_index, const S& start)
  requires requires { {g(start)} -> value::value; }
#else
  template<typename T, typename Getter, typename L, typename S, std::enable_if_t<
    vector_space_descriptor<T> and value::index<L> and value::index<S> and
    value::value<typename std::invoke_result<Getter, const S&>::type>, int> = 0>
  constexpr auto
  to_euclidean_element(const T& t, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
  {
    if constexpr (static_vector_space_descriptor<T> and value::fixed<L>)
      static_assert(value::to_number(euclidean_local_index) < euclidean_dimension_size_of_v<T>);

    if constexpr (euclidean_vector_space_descriptor<T>) return g(start + euclidean_local_index);
    else return interface::vector_space_traits<T>::to_euclidean_component(t, g, euclidean_local_index, start);
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_TO_EUCLIDEAN_ELEMENT_HPP
