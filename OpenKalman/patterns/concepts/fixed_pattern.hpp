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
 * \brief Definition for \ref patterns::fixed_pattern.
 */

#ifndef OPENKALMAN_COORDINATE_FIXED_PATTERN_HPP
#define OPENKALMAN_COORDINATE_FIXED_PATTERN_HPP

#include "collections/collections.hpp"
#include "sized_pattern.hpp"
#include "patterns/traits/dimension_of.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief A \ref patterns::pattern for which the \ref patterns::dimension_of "dimension" is fixed at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed_pattern =
#else
  constexpr bool fixed_pattern =
#endif
    sized_pattern<T> and
    values::fixed_value_compares_with<dimension_of<T>, stdex::dynamic_extent, &stdex::is_neq>;

}

#endif
