/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::make_descriptor_range.
 */

#ifndef OPENKALMAN_PATTERNS_MAKE_DESCRIPTOR_RANGE_HPP
#define OPENKALMAN_PATTERNS_MAKE_DESCRIPTOR_RANGE_HPP

#include "patterns/concepts/descriptor.hpp"
#include "patterns/concepts/descriptor_collection.hpp"
#include "patterns/descriptors/Dimensions.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief Make a \ref descriptor_collection from a list of \ref descriptor "descriptors"
   * \details The result will be a std::ranges::random_access_range<T>.
   * To create a tuple-like structure instead, you should construct a std::tuple.
   */
#ifdef __cpp_concepts
  template<descriptor...Args>
  constexpr descriptor_collection auto
#else
  template<typename...Args, std::enable_if_t<(... and descriptor<Args>), int> = 0>
  constexpr auto
#endif
  make_descriptor_range(Args&&...args)
  {
    if constexpr (sizeof...(Args) == 0)
      return stdex::ranges::views::empty<Dimensions<1>>;
    else if constexpr (sizeof...(Args) == 1)
      return stdex::ranges::views::single(std::forward<Args>(args)...);
    else
      return std::array {static_cast<stdex::common_reference_t<Args...>>(std::forward<Args>(args))...};
  }


}


#endif
