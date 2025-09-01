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
 * \brief Definition for \ref make_constant_diagonal_from_descriptors.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_DIAGONAL_FROM_DESCRIPTORS_HPP
#define OPENKALMAN_MAKE_CONSTANT_DIAGONAL_FROM_DESCRIPTORS_HPP

#include <vector>
#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/pattern_tuple.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Make a constant diagonal from a constant and a set of \ref coordinates::pattern objects.
   */
  template<typename T, typename C, typename Descriptors>
  static constexpr decltype(auto)
  make_constant_diagonal_from_descriptors(C&& c, Descriptors&& descriptors)
  {
    if constexpr (coordinates::pattern_tuple<Descriptors>)
    {
      auto new_descriptors = std::tuple_cat(
        std::tuple(internal::smallest_pattern<scalar_type_of_t<T>>(
          std::get<0>(std::forward<Descriptors>(descriptors)), std::get<1>(std::forward<Descriptors>(descriptors)))),
        internal::tuple_slice<2, collections::size_of_v<Descriptors>>(descriptors));
      return make_constant<T>(std::forward<C>(c), new_descriptors);
    }
    else
    {
#if __cpp_lib_containers_ranges >= 202202L and __cpp_lib_ranges_concat >= 202403L
      auto new_indices = std::views::concat(
        internal::smallest_pattern<scalar_type_of_t<T>>(std::ranges::views::take(indices, 2)),
        indices | std::ranges::views::drop(2));
#else
      auto it = stdcompat::ranges::begin(descriptors);
      auto new_descriptors = std::vector<std::decay_t<decltype(*it)>>{};
      auto i0 = it;
      auto i1 = ++it;
      if (i1 == end(descriptors))
      {
        new_descriptors.emplace_back(coordinates::Axis{});
      }
      else if (i0 != end(descriptors))
      {
        auto d0 = internal::smallest_pattern<scalar_type_of_t<T>>(*i0, *i1);
        new_descriptors.emplace_back(d0);
        std::copy(++it, stdcompat::ranges::end(descriptors), ++stdcompat::ranges::begin(new_descriptors));
      }
#endif
      return make_constant<T>(std::forward<C>(c), new_descriptors);
    }
  }

}

#endif
