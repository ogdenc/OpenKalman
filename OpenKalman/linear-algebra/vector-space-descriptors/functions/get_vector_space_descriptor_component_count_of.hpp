/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_vector_space_descriptor_component_count_of.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COMPONENT_COUNT_OF_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COMPONENT_COUNT_OF_HPP

#include <tuple>
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_collection.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_collection_of.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Get the number of components of \ref vector_space_descriptor T
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, std::enable_if_t<vector_space_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_vector_space_descriptor_component_count_of(const T& t)
  {
    using C = std::decay_t<decltype(descriptor::get_collection_of(t))>;
    if constexpr (descriptor::static_vector_space_descriptor_collection<C>)
    {
      return std::tuple_size_v<C>;
    }
    else
    {
#ifdef __cpp_lib_ranges
      return std::ranges::size(descriptor::get_collection_of(t));
#else
      using std::size;
      return size(descriptor::get_collection_of(t));
#endif
    }
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COMPONENT_COUNT_OF_HPP
