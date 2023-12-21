/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref truncate_indices.
 */

#ifndef OPENKALMAN_ELEMENT_FUNCTIONS_HPP
#define OPENKALMAN_ELEMENT_FUNCTIONS_HPP

#ifdef __cpp_lib_ranges
#include<ranges>
//#else
#include<algorithm>
#endif

namespace OpenKalman::internal
{
  template<std::size_t N, typename Indices>
  constexpr decltype(auto) truncate_indices(const Indices& indices)
  {
    if constexpr (static_range_size_v<Indices> != dynamic_size and N != dynamic_size and static_range_size_v<Indices> > N)
    {
#ifdef __cpp_lib_ranges
      if (std::ranges::any_of(std::views::drop(indices, N), [](const auto& x){ return x != 0; }))
        throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
      return std::views::take(indices, N);
#else
      auto ad = indices.begin();
      std::advance(ad, N);
      if (std::any_of(ad, indices.end(), [](const auto& x){ return x != 0; }))
        throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
      std::array<std::size_t, N> ret;
      std::copy_n(indices.begin(), N, ret.begin());
      return ret;
#endif
    }
    else return indices;
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
