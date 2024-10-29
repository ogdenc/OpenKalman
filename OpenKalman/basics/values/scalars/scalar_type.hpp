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
 * \brief Definition for \scalar_type.
 */

#ifndef OPENKALMAN_SCALAR_TYPE_HPP
#define OPENKALMAN_SCALAR_TYPE_HPP

#include <limits>
#include <type_traits>
#include "basics/values/scalars/complex_number.hpp"

namespace OpenKalman
{
  /**
   * \brief T is a scalar type.
   * \details T can be any arithmetic, complex, or custom scalar type in which certain traits in
   * interface::scalar_traits are defined and typical math operations (+, -, *, /, and ==) are also defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar_type =
#else
  constexpr bool scalar_type =
#endif
    std::numeric_limits<std::decay_t<T>>::is_specialized or complex_number<T>;


} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_TYPE_HPP
