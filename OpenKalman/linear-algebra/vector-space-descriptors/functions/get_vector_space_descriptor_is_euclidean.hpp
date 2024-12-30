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
 * \brief Definition for \ref get_vector_space_descriptor_is_euclidean.
 */

#ifndef OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_IS_EUCLIDEAN_HPP
#define OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_IS_EUCLIDEAN_HPP

#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief Determine, at runtime, whether \ref vector_space_descriptor T is untyped.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, std::enable_if_t<vector_space_descriptor<T>, int> = 0>
#endif
  constexpr bool
  get_vector_space_descriptor_is_euclidean(const T& t)
  {
    return interface::vector_space_traits<T>::is_euclidean(t);
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_IS_EUCLIDEAN_HPP
