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
 * \brief Definition for \ref get_euclidean_dimension_size_of.
 */

#ifndef OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
#define OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP

#include <type_traits>


namespace OpenKalman
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
    if constexpr (fixed_vector_space_descriptor<T>) return euclidean_dimension_size_of_v<T>;
    else
    {
      dynamic_vector_space_descriptor_traits ret{t};
      return ret.get_euclidean_size();
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_GET_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
