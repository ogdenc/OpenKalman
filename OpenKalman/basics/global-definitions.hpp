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

namespace OpenKalman
{

  // -------------- //
  //  TriangleType  //
  // -------------- //

  /**
   * \brief The storage order of a matrix or array.
   * \todo Add traits to determine this from native matrices.
   */
  enum struct ElementOrder {
    column_major, ///< Column-major order.
    row_major, ///< Row-major order.
  };


  /**
   * \brief The type of a triangular matrix, either lower, upper, or diagonal.
   */
  enum struct TriangleType {
    lower, ///< The lower-left triangle.
    upper, ///< The upper-right triangle.
    diagonal ///< The diagonal elements of the matrix.
  };


  namespace internal
  {
    /**
     * \internal
     * \brief A constexpr square root function.
     * \tparam Scalar The scalar type.
     * \param x The operand.
     * \return The square root of x.
     */
    template<typename Scalar>
  # ifdef __cpp_consteval
    consteval
  # else
    constexpr
  # endif
    Scalar constexpr_sqrt(Scalar x)
    {
      if constexpr(std::is_integral_v<Scalar>)
      {
        Scalar lo = 0;
        Scalar hi = x / 2 + 1;
        while (lo != hi)
        {
          const Scalar mid = (lo + hi + 1) / 2;
          if (x / mid < mid) hi = mid - 1;
          else lo = mid;
        }
        return lo;
      }
      else
      {
        Scalar cur = 0.5 * x;
        Scalar old = 0.0;
        while (cur != old)
        {
          old = cur;
          cur = 0.5 * (old + x / old);
        }
        return cur;
      }
    }


    /**
     * Compile time power.
     */
    template<typename Scalar>
  # ifdef __cpp_consteval
    consteval
  # else
    constexpr
  # endif
    Scalar constexpr_pow(Scalar a, std::size_t n)
    {
      return n == 0 ? 1 : constexpr_pow(a, n / 2) * constexpr_pow(a, n / 2) * (n % 2 == 0 ?  1 : a);
    }

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_GLOBAL_DEFINITIONS_HPP
