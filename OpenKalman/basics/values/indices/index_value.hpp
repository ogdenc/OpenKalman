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
 * \brief Definition for \ref index_value.
 */

#ifndef OPENKALMAN_INDEX_VALUE_HPP
#define OPENKALMAN_INDEX_VALUE_HPP


namespace OpenKalman
{
  /**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept index_value =
#else
  template<typename T>
  constexpr bool index_value =
#endif
    static_index_value<T> or dynamic_index_value<T>;

} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_VALUE_HPP
