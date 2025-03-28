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
 * \brief Definition for \ref coordinate::descriptor_collection.
 */

#ifndef OPENKALMAN_COORDINATES_GROUP_COLLECTION_HPP
#define OPENKALMAN_COORDINATES_GROUP_COLLECTION_HPP

#include "basics/concepts/collection.hpp"
#include "descriptor_tuple.hpp"
#include "descriptor_range.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief An object describing a collection of /ref coordinate::descriptor objects.
   * \details This will be either a \ref descriptor_tuple or a \ref descriptor_range.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept descriptor_collection =
#else
  constexpr bool descriptor_collection =
#endif
    collection<T> and (descriptor_tuple<T> or descriptor_range<T>);


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_COORDINATES_GROUP_COLLECTION_HPP
