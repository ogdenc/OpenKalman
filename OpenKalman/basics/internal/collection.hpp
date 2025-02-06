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
 * \internal
 * \brief Definition for \ref internal::collection.
 */

#ifndef OPENKALMAN_COLLECTION_HPP
#define OPENKALMAN_COLLECTION_HPP

#include "tuple_like.hpp"
#include "sized_random_access_range.hpp"

namespace OpenKalman::internal
{
  /**
   * \brief An object describing a collection of objects.
   * \details This will be a \ref internal::tuple_like "tuple_like" object or a sized std::ranges::random_access_range.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept collection =
#else
  constexpr bool collection =
#endif
    internal::tuple_like<T> or internal::sized_random_access_range<T>;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
