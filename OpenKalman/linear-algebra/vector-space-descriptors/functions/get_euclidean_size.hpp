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
 * \brief Definition for \ref get_euclidean_size.
 */

#ifndef OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
#define OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP

#include "basics/internal/get_collection_size.hpp"
#include "internal/get_index_table.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_euclidean_index_table.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Get the Euclidean dimension of \ref vector_space_descriptor T
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, std::enable_if_t<vector_space_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_euclidean_size(const T& t)
  {
    if constexpr (atomic_vector_space_descriptor<T>)
    {
      return interface::vector_space_traits<T>::euclidean_size(t);
    }
    else
    {
      return OpenKalman::internal::get_collection_size(descriptor::internal::get_euclidean_index_table(t));
    }
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
