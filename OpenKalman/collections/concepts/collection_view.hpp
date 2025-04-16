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
#include "basics/ranges.hpp"
#endif
#include "collection.hpp"
#include "collections/views/collection_view_interface.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view to a \ref collection which is also a std::ranges:view.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept collection_view = tuple_like<T> and sized_random_access_range<T> and std::ranges::view<T> and
    std::derived_from<std::remove_cvref_t<T>, collection_view_interface<std::remove_cvref_t<T>>>;
#else
  constexpr bool collection_view = tuple_like<T> and sized_random_access_range<T> and ranges::view<T> and
    std::is_base_of_v<collection_view_interface<remove_cvref_t<T>>, remove_cvref_t<T>>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_COLLECTION_VIEW_HPP
