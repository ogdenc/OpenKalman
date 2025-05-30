/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::sized.
 */

#ifndef OPENKALMAN_COLLECTIONS_SIZED_HPP
#define OPENKALMAN_COLLECTIONS_SIZED_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif

namespace OpenKalman::collections
{
  /**
   * \brief An object (std::ranges::sized_range, std::tuple, std::span, etc.) that has a discernible size.
   */
#ifdef __cpp_lib_ranges
  template<typename T>
  concept sized = std::ranges::sized_range<T> or requires { std::tuple_size<std::decay_t<T>>::value; };
#else
  namespace detail_sized
  {
    template<typename T, typename = void>
    struct has_tuple_size : std::false_type {};

    template<typename T>
    struct has_tuple_size<T, std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>> : std::true_type {};

    using namespace std;

    template<typename T>
    constexpr bool sized = ranges::sized_range<remove_cvref_t<T>> or has_tuple_size<T>::value;
  }


  using detail_sized::sized;
#endif

} // OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_SIZED_HPP
