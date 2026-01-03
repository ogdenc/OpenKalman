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

#include "basics/basics.hpp"
#include "collections/functions/get_size.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief An object (std::ranges::sized_range, std::tuple, std::span, etc.) that has a discernible size.
   */
#ifdef __cpp_lib_ranges
  template<typename T>
  concept sized = requires { collections::get_size(std::declval<T>()); };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct sized_impl : std::false_type {};

    template<typename T>
    struct sized_impl<T, std::void_t<decltype(collections::get_size(std::declval<T>()))>> : std::true_type {};
  }


  template<typename T>
  constexpr bool sized = detail::sized_impl<T>::value;
#endif

}

#endif
