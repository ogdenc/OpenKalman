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
 * \brief Definition for \ref dynamic_index_value.
 */

#ifndef OPENKALMAN_DYNAMIC_INDEX_VALUE_HPP
#define OPENKALMAN_DYNAMIC_INDEX_VALUE_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief T is a dynamic index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_index_value = std::integral<std::decay_t<T>> and std::convertible_to<T, const std::size_t&>;
#else
  template<typename T>
  constexpr bool dynamic_index_value = std::is_integral_v<std::decay_t<T>> and std::is_convertible_v<T, const std::size_t&>;
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_DYNAMIC_INDEX_VALUE_HPP
