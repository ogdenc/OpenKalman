/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_ARRAYUTILS_H
#define OPENKALMAN_ARRAYUTILS_H

#include <array>
#include <tuple>
#include <utility>

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


} // namespace OpenKalman::internal

#endif //OPENKALMAN_ARRAYUTILS_H
