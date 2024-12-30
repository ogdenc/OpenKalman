/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_euclidean_dimension_size_of.
 */

#ifndef OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
#define OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"

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
  get_euclidean_dimension_size_of(const T& t)
  {
    return interface::vector_space_traits<T>::euclidean_size(t);
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
