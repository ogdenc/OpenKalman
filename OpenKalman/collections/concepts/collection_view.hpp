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
 * \brief Definition for \ref collections::collection_view.
 */

#ifndef OPENKALMAN_COLLECTIONS_COLLECTION_VIEW_HPP
#define OPENKALMAN_COLLECTIONS_COLLECTION_VIEW_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/view-concepts.hpp"
#endif
#include "uniformly_gettable.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view to a \ref collection which is also a std::ranges:view.
   * \details It may or may not be \ref sized.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept collection_view = std::ranges::view<T> and uniformly_gettable<T> and std::ranges::random_access_range<T>;
#else
  constexpr bool collection_view = ranges::view<T> and uniformly_gettable<T> and ranges::random_access_range<T>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_COLLECTION_VIEW_HPP
