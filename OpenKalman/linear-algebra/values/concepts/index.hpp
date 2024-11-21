/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value::index.
 */

#ifndef OPENKALMAN_VALUE_INDEX_HPP
#define OPENKALMAN_VALUE_INDEX_HPP

#include "static_index.hpp"
#include "dynamic_index.hpp"

namespace OpenKalman::value
{
  /**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept index =
#else
  template<typename T>
  constexpr bool index =
#endif
    value::static_index<T> or value::dynamic_index<T>;

} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_INDEX_HPP
