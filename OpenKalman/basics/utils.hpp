/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_UTILS_H
#define OPENKALMAN_UTILS_H

#include <array>
#include <tuple>
#include <utility>

#if __cplusplus > 201907L
#include <numbers>
#endif

#ifndef __cpp_lib_math_constants
namespace std::numbers
{
  template<typename T>
  inline constexpr T pi_v = 3.141592653589793238462643383279502884L;

  inline constexpr double pi = pi_v<double>;
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

  /// Join two arrays.
  template<typename T, std::size_t size_L, std::size_t size_R>
  constexpr std::array<T, size_L + size_R>
  join(const std::array<T, size_L>& left, const std::array<T, size_R>& right)
  {
    return join_impl(left, right, typename detail::Counter<size_L, 0>::type(),
      typename detail::Counter<size_R, 0>::type());
  }

  /// Prepend an element to an array.
  template<typename T, std::size_t size_R>
  constexpr std::array<const T, 1 + size_R>
  prepend(const T left, const std::array<const T, size_R>& right)
  {
    return prepend_impl(left, right, typename detail::Counter<size_R, 0>::type());
  }


  namespace detail
  {
    template<typename T>
    constexpr T sqrt_impl(T x, T lo, T hi)
    {
      if (lo == hi) return lo;
      const T mid = (lo + hi + 1) / 2;
      if (x / mid < mid) return sqrt_impl<T>(x, lo, mid - 1);
      else return sqrt_impl(x, mid, hi);
    }
  }

  /// A constexpr square root.
  template<typename T>
  constexpr T constexpr_sqrt(T x)
  {
    return detail::sqrt_impl<T>(x, 0, x / 2 + 1);
  }


  namespace detail
  {
    template<std::size_t begin, typename T, std::size_t... I>
    constexpr auto tuple_slice_impl(T&& t, std::index_sequence<I...>)
    {
      return std::forward_as_tuple(std::get<begin + I>(std::forward<T>(t))...);
    }
  }

  /// Return a subset of a tuple, given an index range.
  template<std::size_t index1, std::size_t index2, typename T>
  constexpr auto tuple_slice(T&& t)
  {
    static_assert(index1 <= index2, "Index range is invalid");
    static_assert(index2 <= std::tuple_size_v<std::decay_t<T>>, "Index is out of bounds");
    return detail::tuple_slice_impl<index1>(std::forward<T>(t), std::make_index_sequence<index2 - index1>());
  }


  /// Create a tuple that replicates a value.
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
      return std::tuple_cat(std::make_tuple(t), tuple_replicate<N - 1>(std::forward<T>(t)));
    }
  }

  /// Default split function.
  struct default_split_function
  {
    template<typename, typename, typename Arg>
    static constexpr Arg&& call(Arg&& arg) { return std::forward<Arg>(arg); }
  };

} // namespace OpenKalman::internal

#endif //OPENKALMAN_UTILS_H
