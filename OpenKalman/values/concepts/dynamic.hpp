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
 * \brief Definition for \values::dynamic.
 */

#ifndef OPENKALMAN_VALUES_DYNAMIC_HPP
#define OPENKALMAN_VALUES_DYNAMIC_HPP

#include "fixed.hpp"

namespace OpenKalman::values
{
  /**
   * \brief T is a value that is not \ref fixed at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic =
#else
  constexpr bool dynamic =
#endif
    (not fixed<T>);

}

#endif
