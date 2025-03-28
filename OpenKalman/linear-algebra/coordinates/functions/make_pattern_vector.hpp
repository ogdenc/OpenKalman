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
 * \brief Definition for \ref coordinate::make_pattern_vector.
 */

#ifndef OPENKALMAN_COORDINATE_MAKE_PATTERN_VECTOR_HPP
#define OPENKALMAN_COORDINATE_MAKE_PATTERN_VECTOR_HPP

#include "values/concepts/value.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief Make a \ref coordinate::pattern comprising a std::vector of \ref coordinate::descriptor "descriptors"
   * \details The resulting type <code>std::vector&lt;component::Any&lt;Scalar&gt;&gt;</code>
   */
#ifdef __cpp_concepts
  template<value::value Scalar = double, descriptor...Args>
#else
  template<typename Scalar = double, typename...Args, std::enable_if_t<
    value::value<Scalar> and (... and descriptor<Args>), int> = 0>
#endif
  constexpr std::vector<Any<Scalar>>
  make_pattern_vector(Args&&...args)
  {
    return {static_cast<Any<Scalar>>(std::forward<Args>(args))...};
  }


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_COORDINATE_MAKE_PATTERN_VECTOR_HPP
