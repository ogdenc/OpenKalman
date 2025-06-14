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
 * \brief Definition for \ref coordinates::descriptor.
 */

#ifndef OPENKALMAN_COORDINATES_DESCRIPTOR_HPP
#define OPENKALMAN_COORDINATES_DESCRIPTOR_HPP

#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief T is an atomic (non-separable or non-composite) grouping of \ref coordinates::pattern objects.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept descriptor =
#else
  constexpr bool descriptor =
#endif
    interface::coordinate_descriptor_traits<std::decay_t<T>>::is_specialized;


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_COORDINATES_DESCRIPTOR_HPP
