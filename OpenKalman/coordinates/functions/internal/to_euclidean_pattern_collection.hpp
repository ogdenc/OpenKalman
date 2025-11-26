/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref coordinates::to_euclidean_pattern_collection.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_PATTERN_COLLECTION_HPP
#define OPENKALMAN_TO_EUCLIDEAN_PATTERN_COLLECTION_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/pattern_collection.hpp"

namespace OpenKalman::coordinates::internal
{
  /**
   * \brief Convert a \ref pattern_collection to its equivalent-sized \ref coordinates::euclidean_pattern_collection.
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern_collection T>
  constexpr euclidean_pattern_collection decltype(auto)
#else
  template<typename T, std::enable_if_t<pattern_collection<T>, int> = 0>
  decltype(auto)
#endif
  to_euclidean_pattern_collection(T&& t)
  {
    if constexpr (euclidean_pattern_collection<T>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<T>, stdex::dynamic_extent, stdex::is_neq>)
    {
      return collections::apply(
        [](auto&&...ds){ return std::tuple {get_dimension(std::forward<decltype(ds)>(ds))...}; },
        std::forward<T>(t));
    }
    else
    {
      return stdex::ranges::views::transform(
        collections::views::all(std::forward<T>(t)),
        [](auto&& d){ return get_dimension(std::forward<decltype(d)>(d)); });
    }
  }


}


#endif
