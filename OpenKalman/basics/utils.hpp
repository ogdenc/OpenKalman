/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Defines various utilities used in OpenKalman.
 */

#ifndef OPENKALMAN_UTILS_HPP
#define OPENKALMAN_UTILS_HPP

#include <array>
#include <tuple>
#include <utility>

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
  namespace detail
  {
    template<std::size_t ... N>
    struct num_tuple {};

    template<std::size_t i, typename T>
    struct Append {};

    template<std::size_t i, std::size_t ... Ns>
    struct Append<i, num_tuple<Ns...>>
    {
      using type = num_tuple<i, Ns...>;
    };

    template<std::size_t N, std::size_t i>
    struct Counter
    {
      using type = typename Append<i, typename Counter<N, i + 1>::type>::type;
    };


    template<std::size_t N>
    struct Counter<N, N>
    {
      using type = num_tuple<>;
    };

    template<typename T, std::size_t size_L, std::size_t size_R, std::size_t ... Ls, std::size_t ... Rs>
    constexpr std::array<T, size_L + size_R>
    join_impl(
      const std::array<T, size_L>& left,
      const std::array<T, size_R>& right,
      num_tuple<Ls...>,
      num_tuple<Rs...>)
    {
      return {left[Ls]..., right[Rs]...};
    };

    template<typename T, std::size_t size_R, std::size_t ... Rs>
    constexpr std::array<const T, 1 + size_R>
    prepend_impl(const T left, const std::array<const T, size_R>& right, num_tuple<Rs...>)
    {
      return {left, right[Rs]...};
    };

  } // namespace detail

  /**
   * \internal
   * \brief Joins two arrays.
   * \tparam T The type of the array elements.
   * \tparam size_L The size of the left array.
   * \tparam size_R The size of the right array.
   * \param left The left array.
   * \param right The right array.
   * \return The left array concatenated with the right array.
   */
  template<typename T, std::size_t size_L, std::size_t size_R>
  constexpr std::array<T, size_L + size_R>
  join(const std::array<T, size_L>& left, const std::array<T, size_R>& right)
  {
    return join_impl(left, right, typename detail::Counter<size_L, 0>::type(),
      typename detail::Counter<size_R, 0>::type());
  }

  /**
   * \internal
   * \brief Prepends an element to an array.
   * \tparam T The type of the array elements.
   * \tparam size_R The size of the array to which the element is to be prepended.
   * \param left The element to be prepended.
   * \param right The array to which the element is to be prepended.
   * \return An array with left prepended to right.
   */
  template<typename T, std::size_t size_R>
  constexpr std::array<const T, 1 + size_R>
  prepend(const T left, const std::array<const T, size_R>& right)
  {
    return prepend_impl(left, right, typename detail::Counter<size_R, 0>::type());
  }


  /**
   * \internal
   * \brief A constexpr square root function.
   * \tparam Scalar The scalar type.
   * \param x The operand.
   * \return The square root of x.
   */
  template<typename Scalar>
#ifdef __cpp_consteval
  consteval
#else
  constexpr
#endif
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
#ifdef __cpp_consteval
  consteval
#else
  constexpr
#endif
  Scalar constexpr_pow(Scalar a, std::size_t n)
  {
    return n == 0 ? 1 : constexpr_pow(a, n / 2) * constexpr_pow(a, n / 2) * (n % 2 == 0 ?  1 : a);
  }


  namespace detail
  {
    template<std::size_t begin, typename T, std::size_t... I>
    constexpr auto tuple_slice_impl(T&& t, std::index_sequence<I...>)
    {
      return std::forward_as_tuple(std::get<begin + I>(std::forward<T>(t))...);
    }
  }


  /**
   * \internal
   * \brief Takes a slice of a tuple, given an index range.
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   * \tparam T The tuple type.
   * \param t The tuple.
   * \return The tuple slice.
   */
  template<std::size_t index1, std::size_t index2, typename T>
  constexpr auto tuple_slice(T&& t)
  {
    static_assert(index1 <= index2, "Index range is invalid");
    static_assert(index2 <= std::tuple_size_v<std::decay_t<T>>, "Index is out of bounds");
    return detail::tuple_slice_impl<index1>(std::forward<T>(t), std::make_index_sequence<index2 - index1>());
  }


  /**
   * \internal
   * \brief Creates a tuple that replicates a value N number of times.
   * \tparam N The number of times to replicate.
   * \tparam T The type of the tuple element to replicate.
   * \param t The tuple element to replicate.
   * \return A tuple containing N copies of t.
   */
  template<std::size_t N, typename T>
  constexpr auto tuple_replicate(T&& t)
  {
    if constexpr (N == 0)
    {
      return std::tuple {};
    }
    else if constexpr (N == 1)
    {
      return std::make_tuple(std::forward<T>(t));
    }
    else
    {
      auto r = std::make_tuple(t);
      auto rs = tuple_replicate<N - 1>(std::forward<T>(t));
      return std::tuple_cat(std::move(r), std::move(rs));
    }
  }


  /**
   * \internal
   * \brief The default wrapper function object for matrix splitting operations.
   */
  struct default_split_function
  {
    /**
     * \internal
     * \brief The identity function.
     * \return The input, unchanged.
     */
    template<typename, typename, typename T>
    static constexpr T&& call(T&& t) { return std::forward<T>(t); }
  };

} // namespace OpenKalman::internal

#endif //OPENKALMAN_UTILS_HPP
