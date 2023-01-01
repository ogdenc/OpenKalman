/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Global definitions for OpenKalman.
 */

#ifndef OPENKALMAN_GLOBAL_DEFINITIONS_HPP
#define OPENKALMAN_GLOBAL_DEFINITIONS_HPP

#include <type_traits>
#include <cstdint>
#include <limits>

namespace OpenKalman
{

  /**
   * \brief A constant indicating that the relevant dimension of a matrix has a size that is dynamic.
   * \details A dynamic dimension can be set, or change, during runtime and is not known at compile time.
   */
#ifndef __cpp_lib_span

  static constexpr std::size_t dynamic_size = std::numeric_limits<std::size_t>::max();
#endif


  /**
   * \brief The type of a triangular matrix.
   */
  enum struct TriangleType : int {
    lower, ///< A lower-left triangular matrix.
    upper, ///< An upper-right triangular matrix.
    diagonal, ///< A diagonal matrix (both a lower-left and an upper-right triangular matrix).
    none, ///< Neither upper, lower, or diagonal.
  };


  /**
   * \brief The likelihood, at compile time, that a particular property applies.
   */
  enum struct Likelihood : bool {
    maybe = false, ///< It is not known at compile time whether the property applies, but it's not ruled out.
    definitely = true, ///< The property is known at compile time to apply.
  };

  constexpr Likelihood operator!(Likelihood x)
  {
    return x == Likelihood::definitely ? Likelihood::maybe : Likelihood::definitely;
  }

} // namespace OpenKalman

#endif //OPENKALMAN_GLOBAL_DEFINITIONS_HPP
