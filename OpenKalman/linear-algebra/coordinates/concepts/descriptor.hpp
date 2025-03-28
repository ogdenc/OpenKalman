/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinate::descriptor.
 */

#ifndef OPENKALMAN_COORDINATES_GROUP_HPP
#define OPENKALMAN_COORDINATES_GROUP_HPP

#include "values/concepts/index.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief T is an atomic (non-separable or non-composite) grouping of \ref coordinate::pattern objects.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept descriptor =
#else
  constexpr bool descriptor =
#endif
    interface::coordinate_descriptor_traits<std::decay_t<T>>::is_specialized or value::index<T>;

} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_COORDINATES_GROUP_HPP
