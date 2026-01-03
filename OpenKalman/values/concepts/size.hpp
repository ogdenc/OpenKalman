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
 * \brief Definition for \ref values::size.
 */

#ifndef OPENKALMAN_VALUES_SIZE_HPP
#define OPENKALMAN_VALUES_SIZE_HPP

#include <type_traits>
#include "index.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A type reflecting an unbound \ref values::size "size".
   */
  struct unbounded_size_t
  {
    friend constexpr bool operator==(unbounded_size_t, unbounded_size_t) noexcept { return true; }

#ifdef __cpp_concepts
    template<index T>
#else
    template<typename T, std::enable_if_t<index<T>, int> = 0>
#endif
    friend constexpr bool operator==(unbounded_size_t, const T&) noexcept { return false; }

#ifndef __cpp_lib_three_way_comparison
#ifdef __cpp_concepts
    template<index T>
#else
    template<typename T, std::enable_if_t<index<T>, int> = 0>
#endif
    friend constexpr bool operator==(const T&, unbounded_size_t) noexcept { return false; }
#endif


#ifndef __cpp_concepts
    // So that unbounded_size can be used in an auto template parameter
    constexpr operator std::size_t() const { return stdex::dynamic_extent; }
#endif
  };




  /**
   * \brief An instance of unbounded_size_t;
   */
  inline constexpr unbounded_size_t unbounded_size {};


  /**
   * \brief T is either an \ref values::index "index" representing a size, or \ref unbounded_size_t, which indicates that the size is unbounded.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept size =
#else
  template<typename T>
  constexpr bool size =
#endif
    index<T> or stdex::same_as<std::decay_t<T>, unbounded_size_t>;


}

#endif
