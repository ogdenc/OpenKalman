/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to standard c++ library concepts.
 */

#ifndef OPENKALMAN_COMPATIBILITY_OBJECT_CONCEPTS_HPP
#define OPENKALMAN_COMPATIBILITY_OBJECT_CONCEPTS_HPP

#include "core-concepts.hpp"
#include "comparison.hpp"
#include "common.hpp"

namespace OpenKalman::stdex
{
#ifdef __cpp_lib_concepts
  using std::movable;
  using std::copyable;
  using std::semiregular;
  using std::regular;
#else
  template<typename T>
  inline constexpr bool
  movable =
    std::is_object_v<T> and
    std::is_move_constructible_v<T> and
    std::is_assignable_v<T&, T> and
    swappable<T>;


  template<typename T>
  inline constexpr bool
  copyable =
    copy_constructible<T> and
    movable<T> and
    std::is_assignable_v<T&, T&> and
    std::is_assignable_v<T&, const T&> and
    std::is_assignable_v<T&, const T>;


  template<typename T>
  inline constexpr bool
  semiregular = copyable<T> and default_initializable<T>;


  template<typename T>
  inline constexpr bool
  regular = semiregular<T> and equality_comparable<T>;

#endif

}

#endif