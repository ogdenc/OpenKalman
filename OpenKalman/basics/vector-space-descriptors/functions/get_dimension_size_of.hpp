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
 * \brief Definition for \ref get_dimension_size_of.
 */

#ifndef OPENKALMAN_GET_DIMENSION_SIZE_OF_HPP
#define OPENKALMAN_GET_DIMENSION_SIZE_OF_HPP


namespace OpenKalman
{
  /**
   * \brief Get the dimension of \ref vector_space_descriptor T
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
  constexpr index_value auto
#else
  template<typename T, std::enable_if_t<vector_space_descriptor<T>, int> = 0>
  constexpr auto
#endif
  get_dimension_size_of(const T& t)
  {
    if constexpr (static_vector_space_descriptor<T>)
    {
      return dimension_size_of<T> {};
    }
    else
    {
      dynamic_vector_space_descriptor_traits ret{t};
      return ret.get_size();
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_GET_DIMENSION_SIZE_OF_HPP
