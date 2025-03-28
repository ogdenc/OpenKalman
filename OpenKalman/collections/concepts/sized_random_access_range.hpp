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
 * \brief Definition for \ref collections::sized_random_access_range.
 */

#ifndef OPENKALMAN_COLLECTIONS_SIZED_RANDOM_ACCESS_RANGE_HPP
#define OPENKALMAN_COLLECTIONS_SIZED_RANDOM_ACCESS_RANGE_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif

namespace OpenKalman::collections
{
  /**
   * \brief A \ref std::ranges::sized_range "sized" \ref std::ranges::random_access_range "random access range".
   */
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_remove_cvref)
  template<typename T>
  concept sized_random_access_range =
    std::ranges::random_access_range<std::remove_cvref_t<T>> and std::ranges::sized_range<std::remove_cvref_t<T>>;
#else
  namespace detail_sized_random_access_range
  {
    using namespace std;

    template<typename T>
    constexpr bool sized_random_access_range =
      ranges::random_access_range<remove_cvref_t<T>> and ranges::sized_range<remove_cvref_t<T>>;
  }

  using detail_sized_random_access_range::sized_random_access_range;
#endif

} // OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_SIZED_RANDOM_ACCESS_RANGE_HPP
