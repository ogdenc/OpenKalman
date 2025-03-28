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
 * \internal
 * \file
 * \brief Definition for \ref remove_trailing_1D_descriptors.
 */

#ifndef OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP
#define OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP

#include <type_traits>
#include <algorithm>
#if defined(__cpp_lib_ranges)
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif


namespace OpenKalman::internal
{

  namespace detail
  {
    template<typename...Ds>
    constexpr auto
    remove_trailing_1D_descriptors_tup(Ds&&...ds)
    {
      constexpr auto N = sizeof...(Ds);
      if constexpr (N == 0)
      {
        return std::tuple {};
      }
      else if constexpr (compares_with<std::tuple_element_t<N - 1, std::tuple<Ds...>>, coordinate::Axis>)
      {
        return std::apply(
          [](auto&&...ds2){ return remove_trailing_1D_descriptors_tup(std::forward<decltype(ds2)>(ds2)...); },
          tuple_slice<0, N - 1>(std::forward_as_tuple(std::forward<Ds>(ds)...)));
      }
      else
      {
        return std::tuple {std::forward<Ds>(ds)...};
      }
    }
  } // namespace detail


  /**
   * \internal
   * \brief Remove any trailing, one-dimensional \ref coordinate::pattern objects.
   * \return A \ref pattern_collection containing the resulting, potentially shortened, list of vector space descriptors
   */
#ifdef __cpp_concepts
  template<pattern_collection Descriptors>
#else
  template<typename Descriptors, std::enable_if_t<pattern_collection<Descriptors>, int> = 0>
#endif
  constexpr auto
  remove_trailing_1D_descriptors(Descriptors&& descriptors)
  {
    if constexpr (pattern_tuple<Descriptors>)
    {
      return std::apply(
        [](auto&&...ds){ return detail::remove_trailing_1D_descriptors_tup(std::forward<decltype(ds)>(ds)...); },
        std::forward<Descriptors>(descriptors));
    }
    else
    {
#ifdef __cpp_lib_ranges
      auto n = std::ranges::partition_point(descriptors, [](const auto& x){ return x != coordinate::Axis{}; });
#if __cpp_lib_ranges >= 202202L
      return descriptors | std::ranges::views::take(n);
#endif
      return std::ranges::views::take(descriptors, n);
#else
      auto it = ranges::begin(descriptors);
      for (auto d = ranges::begin(descriptors); d != ranges::end(descriptors); ++descriptors)
      {
        if (*d != coordinate::Axis{}) it = d;
      }
      std::vector<std::decay_t<decltype(*it)>> ret {};
      std::copy(ranges::begin(descriptors), it, ranges::begin(ret));
      return ret;
#endif
    }
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP
