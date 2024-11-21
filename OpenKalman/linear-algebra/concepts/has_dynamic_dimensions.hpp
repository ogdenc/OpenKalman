/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref has_dynamic_dimensions.
 */

#ifndef OPENKALMAN_HAS_DYNAMIC_DIMENSIONS_HPP
#define OPENKALMAN_HAS_DYNAMIC_DIMENSIONS_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that T has at least one index with dynamic dimensions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_dynamic_dimensions =
#else
  constexpr bool has_dynamic_dimensions =
#endif
    (dynamic_index_count_v<T> == dynamic_size) or (dynamic_index_count_v<T> > 0);


} // namespace OpenKalman

#endif //OPENKALMAN_HAS_DYNAMIC_DIMENSIONS_HPP
