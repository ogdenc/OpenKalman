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
 * \brief Definition for \ref values::size.
 */

#ifndef OPENKALMAN_VALUES_SIZE_HPP
#define OPENKALMAN_VALUES_SIZE_HPP

#include <type_traits>
#include "index.hpp"

namespace OpenKalman::values
{
  /**
   * \brief T is either an \ref values::index "index" representing a size, or void which represents that there is no size.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept size =
#else
  template<typename T>
  constexpr bool size =
#endif
    index<T> or std::is_same_v<std::decay_t<T>, stdcompat::unreachable_sentinel_t>;

}

#endif
