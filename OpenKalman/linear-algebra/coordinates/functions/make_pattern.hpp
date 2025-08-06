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
 * \brief Definition for \ref coordinates::make_pattern.
 */

#ifndef OPENKALMAN_COORDINATES_MAKE_PATTERN_HPP
#define OPENKALMAN_COORDINATES_MAKE_PATTERN_HPP

#include "values/concepts/value.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Make a \ref coordinates::pattern comprising a \ref collections::collection "collection" of \ref coordinates::descriptor "descriptors"
   * \tparam Scalar The scalar type, if all arguments are dynamic
   */
#ifdef __cpp_concepts
  template<values::value Scalar = double, descriptor...Args>
  constexpr pattern decltype(auto)
#else
  template<typename Scalar = double, typename...Args, std::enable_if_t<
    values::value<Scalar> and (... and descriptor<Args>), int> = 0>
  constexpr decltype(auto)
#endif
  make_pattern(Args&&...args)
  {
    if constexpr (sizeof...(Args) == 1)
      return (std::forward<Args>(args),...);
    else if constexpr ((... and fixed_pattern<Args>))
      return std::tuple {std::forward<Args>(args)...};
    else
      return std::vector {static_cast<Any<Scalar>>(std::forward<Args>(args))...};
  }


} // namespace OpenKalman::coordinates


#endif
