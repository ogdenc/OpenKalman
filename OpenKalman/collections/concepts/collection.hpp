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
 * \brief Definition for \ref collections::collection.
 */

#ifndef OPENKALMAN_COLLECTIONS_COLLECTION_HPP
#define OPENKALMAN_COLLECTIONS_COLLECTION_HPP

#include "uniformly_gettable.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief An object describing a collection of objects.
   * \details This is either a \ref collections::uniformly_gettable "uniforly gettable" object
   * or a std::ranges::random_access_range. It need not be \ref sized.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept collection =
#else
  constexpr bool collection =
#endif
    stdex::ranges::random_access_range<T> or uniformly_gettable<T>;


}

#endif
