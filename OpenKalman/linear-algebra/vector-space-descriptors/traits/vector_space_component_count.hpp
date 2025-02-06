/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "basics/internal/collection_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_component_collection.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief The number of atomic component parts of a set of \ref vector_space_descriptor.
   * \details The associated static member <code>value</code> is the number of atomic component parts,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T>
#endif
  struct vector_space_component_count
    : OpenKalman::internal::collection_size_of<decltype(descriptor::internal::get_component_collection(std::declval<T>()))> {};


  /**
   * \brief Helper template for \ref vector_space_component_count.
   */
  template<typename T>
  constexpr auto vector_space_component_count_v = vector_space_component_count<T>::value;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_VECTOR_SPACE_COMPONENT_COUNT_HPP
