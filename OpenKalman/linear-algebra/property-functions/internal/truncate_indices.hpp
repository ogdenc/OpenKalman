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

#include <algorithm>
#include <stdexcept>
#include <vector>
#if defined(__cpp_lib_ranges)
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif

namespace OpenKalman::internal
{
#if __cpp_lib_ranges >= 202202L
  template<std::ranges::input_range Indices, values::index MinCount>
  constexpr decltype(auto) truncate_indices(const Indices& indices, const MinCount& min_count)
  {
    auto n {static_cast<std::ranges::range_difference_t<Indices>>(min_count)};
    if (std::ranges::any_of(indices | std::ranges::views::drop(n), [](const auto& x){ return x != 0; }))
      throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
    return indices | std::ranges::views::take(n);
  }
#else
  template<typename Indices, typename MinCount, std::enable_if_t<values::index<MinCount>, int> = 0>
  decltype(auto) truncate_indices(const Indices& indices, const MinCount& min_count)
  {
#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif
    auto n {static_cast<std::size_t>(min_count)};
    auto ad = ranges::begin(indices);
    std::advance(ad, n);
    if (std::any_of(ad, ranges::end(indices), [](const auto& x){ return x != 0; }))
      throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
    if constexpr (values::fixed<MinCount>)
    {
      std::array<std::size_t, MinCount::value> ret;
      std::copy(ranges::begin(indices), ad, ranges::begin(ret));
      return ret;
    }
    else
    {
      std::vector<std::size_t> ret {n};
      std::copy(ranges::begin(indices), ad, ranges::begin(ret));
      return ret;
    }
  }
#endif
} // namespace OpenKalman::internal

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
