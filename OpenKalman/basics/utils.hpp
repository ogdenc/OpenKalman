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

#include <type_traits>
#include <tuple>
#include "basics/global-definitions.hpp"

namespace OpenKalman::internal
{
#if __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<std::size_t index1, typename Arg, std::size_t...Is>
    constexpr auto tuple_slice_impl(Arg&& arg, std::index_sequence<Is...>)
    {
      return std::tuple {std::get<index1 + Is>(std::forward<Arg>(arg))...};
    }
  } // namespace detail
#endif


  /**
   * \internal
   * \brief Takes a slice of a tuple, given an index range.
   * \details The function will copy or move elements from the argument tuple.
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   * \param arg The tuple.
   * \return The tuple slice.
   */
#ifdef __cpp_concepts
  template<std::size_t index1, std::size_t index2, tuple_like Arg> requires
    (index1 <= index2) and (index2 <= std::tuple_size_v<std::remove_reference_t<Arg>>)
#else
  template<std::size_t index1, std::size_t index2, typename Arg, std::enable_if_t<tuple_like<Arg> and
    (index1 <= index2) and (index2 <= std::tuple_size<std::remove_reference_t<Arg>>::value), int> = 0>
#endif
  constexpr auto
  tuple_slice(Arg&& arg)
  {
#if __cpp_generic_lambdas >= 201707L
    return []<std::size_t...Is>(Arg&& arg, std::index_sequence<Is...>){
      return std::tuple {std::get<index1 + Is>(std::forward<Arg>(arg))...};
    }(std::forward<Arg>(arg), std::make_index_sequence<index2 - index1>{});
#else
    return detail::tuple_slice_impl<index1>(std::forward<Arg>(arg), std::make_index_sequence<index2 - index1>{});
#endif
  }


#if __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<std::size_t index1, typename Arg, std::size_t...Is>
    constexpr auto forward_as_tuple_slice_impl(Arg&& arg, std::index_sequence<Is...>)
    {
      return std::forward_as_tuple(std::get<index1 + Is>(std::forward<Arg>(arg))...);
    }
  } // namespace detail
#endif


/**
   * \internal
   * \brief Takes a slice of a tuple, given an index range.
   * \details The function does not copy elements of the argument. Instead, it forwards the elements as rvalue or lvalue references.
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   * \param arg The tuple.
   * \return The tuple slice.
   */
#ifdef __cpp_concepts
  template<std::size_t index1, std::size_t index2, tuple_like Arg> requires
    (index1 <= index2) and (index2 <= std::tuple_size_v<std::remove_reference_t<Arg>>)
#else
  template<std::size_t index1, std::size_t index2, typename Arg, std::enable_if_t<tuple_like<Arg> and
    (index1 <= index2) and (index2 <= std::tuple_size<std::remove_reference_t<Arg>>::value), int> = 0>
#endif
  constexpr auto
  forward_as_tuple_slice(Arg&& arg)
  {
#if __cpp_generic_lambdas >= 201707L
    return []<std::size_t...Is>(Arg&& arg, std::index_sequence<Is...>){
      return std::forward_as_tuple(std::get<index1 + Is>(std::forward<Arg>(arg))...);
    }(std::forward<Arg>(arg), std::make_index_sequence<index2 - index1>{});
#else
    return detail::forward_as_tuple_slice_impl<index1>(std::forward<Arg>(arg), std::make_index_sequence<index2 - index1>{});
#endif
  }


  // -------------- //
  //   fill_tuple   //
  // -------------- //

  /**
   * \internal
   * \brief Creates a tuple that replicates a value N number of times.
   * \details If the argument is an lvalue reference, the result will be a tuple of lvalue references.
   * \tparam N The number of times to replicate.
   * \param arg The tuple element to replicate.
   * \return A tuple containing N copies of t.
   */
  template<std::size_t N, typename Arg>
  constexpr auto
  fill_tuple(Arg&& arg)
  {
    if constexpr (N == 0)
    {
      return std::tuple {};
    }
    else if constexpr (N == 1)
    {
      return std::tuple<Arg> {std::forward<Arg>(arg)};
    }
    else
    {
      auto r = std::tuple<Arg> {arg};
      return std::tuple_cat(std::move(r), fill_tuple<N - 1>(std::forward<Arg>(arg)));
    }
  }


  // ----------------- //
  //   tuple_reverse   //
  // ----------------- //

  namespace detail
  {
    template<std::size_t...Ix>
    constexpr auto
    reverse_index_sequence(std::index_sequence<>){ return std::index_sequence<Ix...>{}; }

    template<std::size_t...Ix, std::size_t I, std::size_t...Is>
    constexpr auto
    reverse_index_sequence(std::index_sequence<I, Is...>)
    {
      return reverse_index_sequence<I, Ix...>(std::index_sequence<Is...>{});
    }


    template<typename Arg, std::size_t...Ix>
    constexpr auto
    tuple_reverse_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      return std::tuple {std::get<Ix>(std::forward<Arg>(arg))...};
    }
  } // namespace detail


  /**
   * \internal
   * \brief Creates a tuple that is the reverse of the tuple-like argument.
   */
#ifdef __cpp_concepts
  template<internal::tuple_like Arg>
#else
  template<typename Arg, std::enable_if_t<internal::tuple_like<Arg>, int> = 0>
#endif
  constexpr auto
  tuple_reverse(Arg&& arg)
  {
    constexpr auto seq = std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{};
    return detail::tuple_reverse_impl(std::forward<Arg>(arg), detail::reverse_index_sequence(seq));
  }


  // ----------------- //
  //   tuple_flatten   //
  // ----------------- //

  namespace detail
  {
    template<typename Arg> constexpr auto tuple_flatten_impl(Arg&&);


    template<typename Arg, std::size_t...Ix>
    constexpr auto
    tuple_flatten_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      if constexpr ((... or internal::tuple_like<std::tuple_element_t<Ix, std::decay_t<Arg>>>))
        return std::tuple_cat(tuple_flatten_impl(std::get<Ix>(std::forward<Arg>(arg)))...);
      else
        return std::forward<Arg>(arg);
    }


    template<typename Arg>
    constexpr auto
    tuple_flatten_impl(Arg&& arg)
    {
      if constexpr (internal::tuple_like<Arg>)
      {
        constexpr auto seq = std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{};
        return tuple_flatten_impl(std::forward<Arg>(arg), seq);
      }
      else return std::tuple {std::forward<Arg>(arg)};
    }
  } // namespace detail


  /**
   * \internal
   * \brief Flatten a tuple-like object.
   */
#ifdef __cpp_concepts
  template<internal::tuple_like Arg>
#else
  template<typename Arg, std::enable_if_t<internal::tuple_like<Arg>, int> = 0>
#endif
  constexpr auto
  tuple_flatten(Arg&& arg)
  {
    return detail::tuple_flatten_impl(std::forward<Arg>(arg));
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_UTILS_HPP
