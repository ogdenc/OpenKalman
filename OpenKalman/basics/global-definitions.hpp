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
#include <complex>

namespace OpenKalman
{

  /**
   * \brief A constant indicating that the relevant dimension of a matrix has a size that is dynamic.
   * \details A dynamic dimension can be set, or change, during runtime and is not known at compile time.
   */
#ifndef __cpp_lib_span

  static constexpr std::size_t dynamic_size = std::numeric_limits<std::size_t>::max();
#endif


  // -------------- //
  //  TriangleType  //
  // -------------- //

  /**
   * \brief The storage order of a matrix or array.
   * \todo Add traits to determine this from native matrices.
   */
  enum struct ElementOrder : int {
    column_major, ///< Column-major order.
    row_major, ///< Row-major order.
  };


  /**
   * \brief The type of a triangular matrix.
   */
  enum struct TriangleType : int {
    none, ///< A non-triangular matrix.
    lower, ///< A lower-left triangular matrix.
    upper, ///< An upper-right triangular matrix.
    diagonal, ///< A diagonal matrix (both a lower-left and an upper-right triangular matrix).
  };


  // -------------------- //
  //    complex_number    //
  // -------------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_complex_number : std::false_type {};


    template<typename T>
    struct is_complex_number<std::complex<T>> : std::true_type {};
  }


  /**
   * \brief T is a std::complex.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex_number = internal::is_complex_number<std::decay_t<T>>::value;
#else
  constexpr bool complex_number = internal::is_complex_number<std::decay_t<T>>::value;
#endif


  // ----------------- //
  //    scalar_type    //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_scalar_type : std::false_type {};


    template<typename T>
    struct is_scalar_type<T, std::enable_if_t<
      std::is_convertible_v<decltype(std::declval<T>() + std::declval<T>()), const std::decay_t<T>> and
      std::is_convertible_v<decltype(std::declval<T>() - std::declval<T>()), const std::decay_t<T>> and
      std::is_convertible_v<decltype(std::declval<T>() * std::declval<T>()), const std::decay_t<T>> and
      std::is_convertible_v<decltype(std::declval<T>() / std::declval<T>()), const std::decay_t<T>>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a scalar type (i.e., is arithmetic, complex, or other number in which + - * and / are defined).
   * \details OpenKalman presumes that elements of an algebraic field may be the entries of a matrix.
   * This includes integral, floating point, and complex values. T need not be an algebraic field, but this concept
   * is designed, conceptually, to capture the idea of a non-commutative division ring that is not necessarily
   * associative under multiplication.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar_type = std::is_arithmetic_v<T> or complex_number<T> or
    requires(T t1, T t2) {
      {t1 + t2} -> std::convertible_to<const std::decay_t<T>>;
      {t1 - t2} -> std::convertible_to<const std::decay_t<T>>;
      {t1 * t2} -> std::convertible_to<const std::decay_t<T>>;
      {t1 / t2} -> std::convertible_to<const std::decay_t<T>>; };
#else
  constexpr bool scalar_type = std::is_arithmetic_v<T> or complex_number<T> or
    detail::is_scalar_type<std::decay_t<T>>::value;
#endif


  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam epsilon_factor A factor to be multiplied by the epsilon
   * \return true if within the rounding tolerance, otherwise false
   */
#ifdef __cpp_concepts
  template<unsigned int epsilon_factor = 2, typename Arg1, typename Arg2> requires
    requires { typename std::numeric_limits<decltype(std::declval<Arg1>() - std::declval<Arg2>())>; }
#else
  template<unsigned int epsilon_factor = 2, typename Arg1, typename Arg2,
    typename = std::void_t<std::numeric_limits<decltype(std::declval<Arg1>() - std::declval<Arg2>())>>>
#endif
  constexpr bool are_within_tolerance(const Arg1& arg1, const Arg2& arg2)
  {
    auto diff = arg1 - arg2;
    using Diff = decltype(diff);
    constexpr auto ep = epsilon_factor * std::numeric_limits<Diff>::epsilon();
    return -static_cast<Diff>(ep) <= diff and diff <= static_cast<Diff>(ep);
  }


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
     * \internal
     * \brief A constexpr power function.
     * \tparam Scalar The scalar type.
     * \param a The operand
     * \param n The power
     * \return a to the power of n.
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
