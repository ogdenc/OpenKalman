/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref reduce function.
 */

#ifndef OPENKALMAN_COUNT_REDUCED_DIMENSIONS_HPP
#define OPENKALMAN_COUNT_REDUCED_DIMENSIONS_HPP

namespace OpenKalman::internal
{
  template<typename T, std::size_t...indices, std::size_t...Is>
  constexpr auto count_reduced_dimensions(const T& t, std::index_sequence<indices...>, std::index_sequence<Is...>)
  {
    if constexpr ((dynamic_dimension<T, indices> or ...))
    {
      return ([](const T& t){
        constexpr auto I = Is;
        return (((I == indices) or ...)) ? get_index_dimension_of<I>(t) : 1;
      }(t) * ... * 1);
    }
    else
    {
      constexpr auto dim = ([]{
        constexpr auto I = Is;
        return (((I == indices) or ...)) ? index_dimension_of_v<T, I> : 1;
      }() * ... * 1);
      return std::integral_constant<std::size_t, dim>{};
    }
  }

}

#endif
