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
 * \brief Definition for \ref value::floating.
 */

#ifndef OPENKALMAN_VALUE_FLOATING_HPP
#define OPENKALMAN_VALUE_FLOATING_HPP

#include <type_traits>
#include "value.hpp"
#include "integral.hpp"
#include "complex.hpp"

namespace OpenKalman::value
{
  /**
   * \brief T is a floating-point value::value.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept floating =
#else
  constexpr bool floating =
#endif
    value::value<T> and (not integral<T>) and (not value::complex<T>);


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_FLOATING_HPP
