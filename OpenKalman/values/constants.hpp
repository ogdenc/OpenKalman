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
 * \brief Global constants relating to \ref collections.
 */

#ifndef OPENKALMAN_VALUES_CONSTANTS_HPP
#define OPENKALMAN_VALUES_CONSTANTS_HPP

#ifdef __cpp_lib_span
#include <span>
#else
#include <limits>
#endif

namespace OpenKalman
{
  /**
   * \brief The applicability of a concept, trait, or restraint.
   * \details Determines whether something is necessarily applicable, or alternatively just permissible, at compile time.
   * Examples:
   * - <code>square_shaped<T, applicability::guaranteed></code> means that T is known at compile time to be square shaped.
   * - <code>square_shaped<T, applicability::permitted></code> means that T <em>could</em> be square shaped,
   * but whether it actually <em>is</em> cannot be determined at compile time.
   */
  enum struct applicability : int {
    guaranteed, ///< The concept, trait, or restraint represents a compile-time guarantee.
    permitted, ///< The concept, trait, or restraint is permitted, but whether it applies is not necessarily known at compile time.
  };


  constexpr applicability operator not (applicability x)
  {
    return x == applicability::guaranteed ? applicability::permitted : applicability::guaranteed;
  }


  constexpr applicability operator and (applicability x, applicability y)
  {
    return x == applicability::guaranteed and y == applicability::guaranteed ? applicability::guaranteed : applicability::permitted;
  }


  constexpr applicability operator or (applicability x, applicability y)
  {
    return x == applicability::guaranteed or y == applicability::guaranteed ? applicability::guaranteed : applicability::permitted;
  }


}

#endif
