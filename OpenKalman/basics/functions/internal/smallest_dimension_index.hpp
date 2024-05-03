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
 * \brief Definition for \ref internal::smallest_dimension_index function.
 */

#ifndef OPENKALMAN_SMALLEST_DIMENSION_INDEX_HPP
#define OPENKALMAN_SMALLEST_DIMENSION_INDEX_HPP

namespace OpenKalman::internal
{
  /**
   * \brief Return the \ref index_value of the index having the smallest dimension
   * \details If the dimensions are the same, this will return index 0.
   * \tparam Arg A matrix
   * \return An \ref index_value
   */
  template<typename Arg>
  constexpr auto
  smallest_dimension_index(const Arg& arg)
  {
    using D0 = vector_space_descriptor_of_t<Arg, 0>;
    using D1 = vector_space_descriptor_of_t<Arg, 1>;
    if constexpr (fixed_vector_space_descriptor<D0> and fixed_vector_space_descriptor<D1>)
    {
      if constexpr (dimension_size_of_v<D1> < dimension_size_of_v<D0>) return std::integral_constant<std::size_t, 1>{};
      else return std::integral_constant<std::size_t, 0>{};
    }
    else
    {
      if (get_index_dimension_of<1>(arg) < get_index_dimension_of<0>(arg)) return 1_uz;
      else return 0_uz;
    }
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_SMALLEST_DIMENSION_INDEX_HPP
