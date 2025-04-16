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
#include "basics/ranges.hpp"
#endif
#include "collection.hpp"
#include "collections/traits/common_collection_type.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename T, typename = void>
    struct has_common_collection_type : std::false_type {};

    template<typename T>
    struct has_common_collection_type<T, std::void_t<typename common_collection_type<T>::type>> : std::true_type {};
  }
  /**
   * \brief A \ref collections::collection "collection" that can be converted into a
   * a \ref collections::collection_view "collection_view" through \ref collections::views::all.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept viewable_collection = collection<T> and
    (std::ranges::viewable_range<T> or requires { typename common_collection_type<T>::type; });
#else
  constexpr bool viewable_collection = collection<T> and
    (ranges::viewable_range<T> or detail::has_common_collection_type<T>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_VIEWABLE_COLLECTION_HPP
