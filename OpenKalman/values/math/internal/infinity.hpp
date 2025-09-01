/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file \internal
 * \brief Definition for \ref values::internal::infinity.
 */

#ifndef OPENKALMAN_VALUES_INFINITY_HPP
#define OPENKALMAN_VALUES_INFINITY_HPP

#include <limits>
#include <stdexcept>
#include "values/concepts/number.hpp"

namespace OpenKalman::values::internal
{
  /**
   * \internal
   * \brief Return +infinity in type T or raise an exception if infinity is not available.
   */
#ifdef __cpp_concepts
  template <values::number T>
#else
  template <typename T, std::enable_if_t<values::number<T>, int> = 0>
#endif
  constexpr std::decay_t<T> infinity()
  {
    using R = std::decay_t<T>;
    if constexpr (std::numeric_limits<R>::has_infinity) return std::numeric_limits<R>::infinity();
    else throw std::domain_error {"Domain error in arithmetic operation: result is infinity"};
  }

}


#endif
