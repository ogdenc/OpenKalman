/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref has_uniform_reduction_indices function.
 */

#ifndef OPENKALMAN_HAS_UNIFORM_REDUCTION_INDICES_HPP
#define OPENKALMAN_HAS_UNIFORM_REDUCTION_INDICES_HPP

namespace OpenKalman::internal
{

  /**
   * \internal
   * \brief Whether an object's specified indices are uniform such that the object can be reduced along those indices.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr bool has_uniform_reduction_indices(std::index_sequence<>) { return true; }


#ifdef __cpp_concepts
  template<indexible Arg, std::size_t I, std::size_t...Is>
#else
  template<typename Arg, std::size_t I, std::size_t...Is, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr bool has_uniform_reduction_indices(std::index_sequence<I, Is...>)
  {
    return (has_uniform_dimension_type<vector_space_descriptor_of_t<Arg, I>> or dynamic_dimension<Arg, I>) and
      (not dimension_size_of_index_is<Arg, I, 0>) and
      ((I != Is) and ...) and (has_uniform_reduction_indices<Arg>(std::index_sequence<Is...>{}));
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_HAS_UNIFORM_REDUCTION_INDICES_HPP
