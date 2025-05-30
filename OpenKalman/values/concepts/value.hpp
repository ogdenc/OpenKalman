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
 * \brief Definition for \values::value.
 */

#ifndef OPENKALMAN_VALUE_SCALAR_CONSTANT_HPP
#define OPENKALMAN_VALUE_SCALAR_CONSTANT_HPP

#include "fixed.hpp"
#include "dynamic.hpp"

namespace OpenKalman::values
{
  /**
   * \brief T is numerical value or is reducible to a numerical value.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept value =
#else
  constexpr bool value =
#endif
    values::fixed<T> or dynamic<T>;


} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUE_SCALAR_CONSTANT_HPP
