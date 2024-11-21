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
 * \brief Definition for \ref value::floating_number.
 */

#ifndef OPENKALMAN_VALUE_FLOATING_SCALAR_HPP
#define OPENKALMAN_VALUE_FLOATING_SCALAR_HPP

#include <type_traits>
#include "complex_number.hpp"
#include "number.hpp"

namespace OpenKalman::value
{
  /**
   * \brief T is a floating-point scalar type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept floating_number =
#else
  constexpr bool floating_number =
#endif
    value::number<T> and not std::is_integral_v<std::decay_t<T>> and not value::complex_number<T>;


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_FLOATING_SCALAR_HPP
