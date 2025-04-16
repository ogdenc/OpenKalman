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
 * \brief Definition for \ref get_collection_size.
 */

#ifndef OPENKALMAN_GET_COLLECTION_SIZE_HPP
#define OPENKALMAN_GET_COLLECTION_SIZE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "collections/concepts/collection.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief Get the size of a \ref collection
   */
#ifdef __cpp_concepts
  template<collection Arg>
#else
  template<typename Arg, std::enable_if_t<collection<Arg>, int> = 0>
#endif
  constexpr auto
  get_collection_size(Arg&& arg)
  {
    if constexpr (size_of_v<Arg> == dynamic_size)
    {
#ifdef __cpp_lib_ranges
      return std::ranges::size(std::forward<Arg>(arg));
#else
      return ranges::size(std::forward<Arg>(arg));
#endif
    }
    else
    {
      return size_of<Arg>{};
    }
  };

} // OpenKalman::collections


#endif //OPENKALMAN_GET_COLLECTION_SIZE_HPP
