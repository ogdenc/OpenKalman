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
#if defined(__cpp_lib_ranges) and not defined (__clang__)
#include <ranges>
#endif

namespace OpenKalman::internal
{
#if defined(__cpp_lib_ranges) and not defined (__clang__)
  template<std::ranges::input_range Indices, value::index MinCount>
  constexpr decltype(auto) truncate_indices(const Indices& indices, const MinCount& min_count)
  {
    auto n {static_cast<std::ranges::range_difference_t<Indices>>(min_count)};
    if (std::ranges::any_of(indices | std::ranges::views::drop(n), [](const auto& x){ return x != 0; }))
      throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
    return indices | std::ranges::views::take(n);
  }
#else
  template<typename Indices, typename MinCount, std::enable_if_t<value::index<MinCount>, int> = 0>
  decltype(auto) truncate_indices(const Indices& indices, const MinCount& min_count)
  {
    using std::begin, std::end;
    auto n {static_cast<std::size_t>(min_count)};
    auto ad = begin(indices);
    std::advance(ad, n);
    if (std::any_of(ad, end(indices), [](const auto& x){ return x != 0; }))
      throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
    if constexpr (value::fixed<MinCount>)
    {
      std::array<std::size_t, MinCount::value> ret;
      std::copy(begin(indices), ad, begin(ret));
      return ret;
    }
    else
    {
      std::vector<std::size_t> ret {n};
      std::copy(begin(indices), ad, begin(ret));
      return ret;
    }
  }
#endif
} // namespace OpenKalman::internal

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
