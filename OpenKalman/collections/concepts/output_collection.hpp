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
 * \brief Definition for \ref collections::output_collection.
 */

#ifndef OPENKALMAN_COLLECTIONS_OUTPUT_COLLECTION_HPP
#define OPENKALMAN_COLLECTIONS_OUTPUT_COLLECTION_HPP

#include "collection.hpp"
#include "uniformly_settable.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref collection that can be modified on an element-by-element basis.
   */
  template<typename C, typename T>
#ifdef __cpp_concepts
  concept output_collection =
#else
  constexpr bool output_collection =
#endif
    collection<C> and (stdex::ranges::output_range<C, T> or uniformly_settable<C, T>);


}

#endif
