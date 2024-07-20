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


  // -------- //
  //   join   //
  // -------- //

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


  // ----------- //
  //   prepend   //
  // ----------- //

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


  // --------------- //
  //   tuple_slice   //
  // --------------- //

  namespace detail
  {
    template<typename...Ts>
    auto make_sub_tuple(Ts&&...ts)
    {
      return std::tuple<Ts...>(std::forward<Ts>(ts)...);
    }

    template<std::size_t index1, typename T, std::size_t...Is>
    constexpr auto tuple_slice_impl(T&& t, std::index_sequence<0, Is...>)
    {
      return make_sub_tuple(std::get<index1>(std::forward<T>(t)), std::get<index1 + Is>(std::forward<T>(t))...);
    }
  } // namespace detail


  /**
   * \internal
   * \brief Takes a slice of a tuple, given an index range.
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   * \tparam T The tuple type.
   * \param t The tuple.
   * \return The tuple slice.
   */
#ifdef __cpp_concepts
  template<std::size_t index1, std::size_t index2, tuple_like T> requires
    (index1 <= index2) and (index2 <= std::tuple_size_v<std::remove_reference_t<T>>)
#else
  template<std::size_t index1, std::size_t index2, typename T, std::enable_if_t<tuple_like<T> and
    (index1 <= index2) and (index2 <= std::tuple_size<std::remove_reference_t<T>>::value), int> = 0>
#endif
  constexpr auto tuple_slice(T&& t)
  {
    if constexpr (index1 == index2) return std::tuple{};
    else return detail::tuple_slice_impl<index1>(std::forward<T>(t), std::make_index_sequence<index2 - index1>{});
  }


  // ------------------- //
  //   tuple_replicate   //
  // ------------------- //

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
      auto r = std::make_tuple(t); //< Make a copy.
      return std::tuple_cat(std::move(r), tuple_replicate<N - 1>(std::forward<T>(t)));
    }
  }


  // -------------------------- //
  //   default_split_function   //
  // -------------------------- //

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
