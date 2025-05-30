/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinates::dynamic_pattern.
 */

#ifndef OPENKALMAN_COORDINATE_DYNAMIC_PATTERN_HPP
#define OPENKALMAN_COORDINATE_DYNAMIC_PATTERN_HPP

#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief A \ref coordinates::pattern for which the \ref coordinates::dimension_of "size" is defined at runtime.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_pattern =
#else
  constexpr bool dynamic_pattern =
#endif
    pattern<T> and (not fixed_pattern<T>);


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_COORDINATE_DYNAMIC_PATTERN_HPP
