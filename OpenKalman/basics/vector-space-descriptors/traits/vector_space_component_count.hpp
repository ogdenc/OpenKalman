/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref vector_space_component_count.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP
#define OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief The number of atomic component parts of a set of \ref vector_space_descriptor.
   * \details The associated static member <code>value</code> is the number of atomic component parts,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct vector_space_component_count : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
  struct vector_space_component_count<T>
#else
  template<typename T>
  struct vector_space_component_count<T, std::enable_if_t<static_vector_space_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, static_vector_space_descriptor_traits<std::decay_t<T>>::component_count> {};


  /**
   * \brief Helper template for \ref vector_space_component_count.
   */
  template<typename T>
  constexpr auto vector_space_component_count_v = vector_space_component_count<T>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP
