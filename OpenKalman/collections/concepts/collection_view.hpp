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

#include "basics/basics.hpp"
#include "collection.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view to a \ref collection which is also a std::ranges:view.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept collection_view =
#else
  constexpr bool collection_view =
#endif
    collection<T> and
    stdex::ranges::view<T>;


}

#endif
