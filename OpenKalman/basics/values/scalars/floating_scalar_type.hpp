/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \floating_scalar_type.
 */

#ifndef OPENKALMAN_FLOATING_SCALAR_TYPE_HPP
#define OPENKALMAN_FLOATING_SCALAR_TYPE_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief T is a floating-point scalar type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept floating_scalar_type =
#else
  constexpr bool floating_scalar_type =
#endif
    scalar_type<T> and not std::is_integral_v<std::decay_t<T>> and not complex_number<T>;


} // namespace OpenKalman

#endif //OPENKALMAN_FLOATING_SCALAR_TYPE_HPP
