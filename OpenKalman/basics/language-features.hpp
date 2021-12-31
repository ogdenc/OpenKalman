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
 * \brief Definitions relating to the availability of c++ language features.
 */

#ifndef OPENKALMAN_LANGUAGE_FEATURES_HPP
#define OPENKALMAN_LANGUAGE_FEATURES_HPP

#include <type_traits>


#ifdef __clang__
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS             true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2           false // clang versions known to cause bugs: 10.0.0
#elif defined(__GNUC__)
#  if __GNUC__ > 100
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS           true
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS_2         true
#  else
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS           false // GCC versions known to cause bugs: 10.1.0
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS_2         false // GCC versions known to cause bugs: 10.1.0
#  endif
#else
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS             true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2           true
#endif


#if __cplusplus > 201907L
#include <numbers>
#endif

#ifndef __cpp_lib_math_constants
// These are re-creations of some of the c++20 standard constants, if they are not already defined.
namespace std::numbers
{
  template<typename T>
  inline constexpr T pi_v = 3.141592653589793238462643383279502884L;

  inline constexpr double pi = pi_v<double>;

  template<typename T>
  inline constexpr T log2e_v = 1.442695040888963407359924681001892137L;

  inline constexpr double log2e = log2e_v<double>;

  template<typename T>
  inline constexpr T sqrt2_v = 1.414213562373095048801688724209698079L;

  inline constexpr double sqrt2 = sqrt2_v<double>;
}
#endif


namespace OpenKalman::internal
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

} // namespace OpenKalman::internal

#endif //OPENKALMAN_LANGUAGE_FEATURES_HPP
