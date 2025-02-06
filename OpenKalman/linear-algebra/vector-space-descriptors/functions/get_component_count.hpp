/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_component_count.
 */

#ifndef OPENKALMAN_GET_COMPONENT_COUNT_HPP
#define OPENKALMAN_GET_COMPONENT_COUNT_HPP

#include "basics/internal/get_collection_size.hpp"
#include "linear-algebra/values/concepts/index.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_component_collection.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Get the number of components of a \ref vector_space_descriptor
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
constexpr value::index auto
#else
  template<typename T, std::enable_if_t<vector_space_descriptor<T>, int> = 0>
  constexpr auto
#endif
  get_component_count(const T& t)
  {
    if constexpr (atomic_vector_space_descriptor<T>)
    {
      return std::integral_constant<std::size_t, 1_uz>{};
    }
    else
    {
      return OpenKalman::internal::get_collection_size(descriptor::internal::get_component_collection(t));
    }

  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_COMPONENT_COUNT_HPP
