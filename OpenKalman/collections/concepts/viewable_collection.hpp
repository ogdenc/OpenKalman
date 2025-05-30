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
 * \brief Definition for \ref collections::viewable_collection.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWABLE_COLLECTION_HPP
#define OPENKALMAN_COLLECTIONS_VIEWABLE_COLLECTION_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/view-concepts.hpp"
#endif
#include "uniform_tuple_like.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A std::range or \ref uniform_tuple_like object that can be converted into
   * a \ref collection_view by passing it to \ref collections::views::all.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept viewable_collection = (std::ranges::random_access_range<T> and
    (std::ranges::view<std::remove_cvref_t<T>> or std::ranges::viewable_range<T>)) or uniform_tuple_like<T>;
#else
  constexpr bool viewable_collection = (ranges::random_access_range<T> and
    (ranges::view<remove_cvref_t<T>> or ranges::viewable_range<T>)) or uniform_tuple_like<T>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_VIEWABLE_COLLECTION_HPP
