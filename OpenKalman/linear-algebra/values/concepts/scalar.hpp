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
 * \brief Definition for \value::scalar.
 */

#ifndef OPENKALMAN_VALUE_SCALAR_CONSTANT_HPP
#define OPENKALMAN_VALUE_SCALAR_CONSTANT_HPP

#include "static_scalar.hpp"
#include "dynamic_scalar.hpp"

namespace OpenKalman::value
{
  /**
   * \brief T is a scalar constant
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar =
#else
  constexpr bool scalar =
#endif
    static_scalar<T> or dynamic_scalar<T>;


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_SCALAR_CONSTANT_HPP
