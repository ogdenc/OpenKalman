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
 * \brief Definition for \ref coordinate::to_euclidean_vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_PATTERN_COLLECTION_HPP
#define OPENKALMAN_TO_EUCLIDEAN_PATTERN_COLLECTION_HPP


namespace OpenKalman::coordinate::internal
{
  /**
   * \brief Convert a \ref pattern_collection to its equivalent-sized \ref coordinate::euclidean_pattern_collection.
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern_collection T>
  constexpr euclidean_pattern_collection decltype(auto)
#else
  template<typename T, std::enable_if_t<pattern_collection<T>, int> = 0>
  decltype(auto)
#endif
  to_euclidean_vector_space_descriptor_collection(T&& t)
  {
    if constexpr (euclidean_pattern_collection<T>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (pattern_tuple<T>)
    {
      return std::apply([](auto&&...ds){
        return std::tuple {get_size(std::forward<decltype(ds)>(ds))...};
        }, std::forward<T>(t));
    }
    else
    {
      std::vector<std::size_t> ret {};
#ifdef __cpp_lib_ranges
      std::ranges::transform(std::ranges::begin(t), t, [](auto&& d){ return get_size(std::forward<decltype(d)>(d)); });
#else
      std::transform(ranges::begin(t), ranges::end(t), ranges::begin(ret), [](auto&& d){ return get_size(std::forward<decltype(d)>(d)); });
#endif
      return ret;
    }
  }


} // namespace OpenKalman::coordinate::internal


#endif //OPENKALMAN_TO_EUCLIDEAN_PATTERN_COLLECTION_HPP
