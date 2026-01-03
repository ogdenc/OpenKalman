/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::pattern.
 */

#ifndef OPENKALMAN_COORDINATE_PATTERN_HPP
#define OPENKALMAN_COORDINATE_PATTERN_HPP

#include "descriptor.hpp"
#include "descriptor_collection.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief An object describing the characteristics (e.g., dimensions, wrapping structure) of an index.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept pattern =
#else
  inline constexpr bool pattern =
#endif
    descriptor<T> or descriptor_collection<T>;


}

#endif
