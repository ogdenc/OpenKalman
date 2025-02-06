/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref value::internal::get_collection_size.
 */

#ifndef OPENKALMAN_GET_COLLECTION_SIZE_HPP
#define OPENKALMAN_GET_COLLECTION_SIZE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#endif
#include "basics/internal/collection.hpp"
#include "basics/internal/collection_size_of.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Get the size of a \ref internal::collection "collection"
   */
#ifdef __cpp_concepts
  template<OpenKalman::internal::collection Arg>
#else
  template<typename Arg, std::enable_if_t<OpenKalman::internal::collection<Arg>, int> = 0>
#endif
  constexpr auto
  get_collection_size(const Arg& arg)
  {
    if constexpr (OpenKalman::internal::collection_size_of_v<Arg> == dynamic_size)
    {
#ifdef __cpp_lib_ranges
      return std::ranges::size(arg);
#else
      using std::size;
      return size(arg);
#endif
    }
    else
    {
      return OpenKalman::internal::collection_size_of<Arg>{};
    }
  };

} // OpenKalman::internal


#endif //OPENKALMAN_GET_COLLECTION_SIZE_HPP
