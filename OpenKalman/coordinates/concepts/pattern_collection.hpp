/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref pattern_collection.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP

#include "collections/concepts/collection.hpp"
#include "pattern_tuple.hpp"
#include "pattern_range.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief An object describing a collection of /ref  coordinates::pattern objects.
   * \details This will be a \ref pattern_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept pattern_collection =
#else
  constexpr bool pattern_collection =
#endif
    collections::collection<T> and (pattern_tuple<T> or pattern_range<T>);


}

#endif
