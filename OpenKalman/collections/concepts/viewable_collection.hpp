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

#include "basics/basics.hpp"
#include "viewable_tuple_like.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A std::range or \ref viewable_tuple_like object that can be converted into
   * a \ref collection_view by passing it to \ref collections::views::all.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept viewable_collection =
#else
  constexpr bool viewable_collection =
#endif
    (stdex::ranges::random_access_range<T> and stdex::ranges::viewable_range<T>) or
    viewable_tuple_like<T>;


}

#endif
