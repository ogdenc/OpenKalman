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
 * \brief Definition for \ref to_euclidean_vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_TO_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP


namespace OpenKalman::internal
{
  /**
   * \brief Convert a \ref vector_space_descriptor_collection to its equivalent-sized \ref euclidean_vector_space_descriptor_collection.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor_collection T>
  constexpr euclidean_vector_space_descriptor_collection decltype(auto)
#else
  template<typename T, std::enable_if_t<vector_space_descriptor_collection<T>, int> = 0>
  constexpr decltype(auto)
#endif
  to_euclidean_vector_space_descriptor_collection(T&& t)
  {
    if constexpr (euclidean_vector_space_descriptor_collection<T>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (vector_space_descriptor_tuple<T>)
    {
      return std::apply([](auto&&...ds){
        return std::tuple {get_dimension_size_of(std::forward<decltype(ds)>(ds))...};
        }, std::forward<T>(t));
    }
    else
    {
      std::vector<std::size_t> ret {};
#ifdef __cpp_lib_ranges
      std::ranges::transform(t.begin(), t, [](auto&& d){ return get_dimension_size_of(std::forward<decltype(d)>(d)); });
#else
      std::transform(t.begin(), t.end(), ret.begin(), [](auto&& d){
        return get_dimension_size_of(std::forward<decltype(d)>(d));
      });
#endif
      return ret;
    }
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_TO_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
