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

#include <type_traits>


namespace OpenKalman
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
    if constexpr (fixed_vector_space_descriptor<T>) return vector_space_component_count_v<T>;
    else
    {
      dynamic_vector_space_descriptor_traits ret{t};
      return ret.get_component_count();
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COMPONENT_COUNT_OF_HPP
