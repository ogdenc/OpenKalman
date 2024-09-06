/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref get_reduced_descriptor function.
 */

#ifndef OPENKALMAN_GET_REDUCED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_GET_REDUCED_VECTOR_SPACE_DESCRIPTOR_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Return a \ref vector_space_descriptor for arg after reducing along index I
   * \tparam I The index along which the reduction takes place
   * \tparam indices The set of reduced indices
   * \tparam Arg The indexible object
   */
#ifdef __cpp_concepts
  template<std::size_t I, std::size_t...indices, indexible Arg>
  constexpr vector_space_descriptor auto
#else
  template<std::size_t I, std::size_t...indices, typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr auto
#endif
  get_reduced_vector_space_descriptor(const Arg& arg)
  {
    if constexpr ((... or (I == indices)))
      return uniform_fixed_vector_space_descriptor_component_of_t<vector_space_descriptor_of_t<Arg, I>>{};
    else
      return get_vector_space_descriptor<I>(arg);
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_GET_REDUCED_VECTOR_SPACE_DESCRIPTOR_HPP
