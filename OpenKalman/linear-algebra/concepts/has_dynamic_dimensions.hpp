/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref has_dynamic_dimensions.
 */

#ifndef OPENKALMAN_HAS_DYNAMIC_DIMENSIONS_HPP
#define OPENKALMAN_HAS_DYNAMIC_DIMENSIONS_HPP

#include "linear-algebra/traits/dynamic_index_count.hpp"

namespace OpenKalman
{
  /**
   * \brief Specifies that T has at least one index with dynamic dimensions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_dynamic_dimensions =
#else
  constexpr bool has_dynamic_dimensions =
#endif
    (dynamic_index_count_v<T> == stdex::dynamic_extent) or (dynamic_index_count_v<T> > 0);


}

#endif
