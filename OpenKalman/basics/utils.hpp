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
      if constexpr (std::is_rvalue_reference_v<Arg&&>)
        return std::tuple {std::get<index1 + Is>(std::forward<Arg>(arg))...};
      else
        return std::forward_as_tuple(std::get<index1 + Is>(std::forward<Arg>(arg))...);
    }
  } // namespace detail
#endif


  /**
   * \internal
   * \brief Takes a slice of a tuple, given an index range.
   * \details If the argument is an lvalue reference, the result will be a tuple of lvalue references.
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
      if constexpr (std::is_rvalue_reference_v<Arg&&>)
        return std::tuple {std::get<index1 + Is>(std::forward<Arg>(arg))...};
      else
        return std::forward_as_tuple(std::get<index1 + Is>(std::forward<Arg>(arg))...);
    }(std::forward<Arg>(arg), std::make_index_sequence<index2 - index1>{});
#else
    return detail::tuple_slice_impl<index1>(std::forward<Arg>(arg), std::make_index_sequence<index2 - index1>{});
#endif
  }


  // ------------------- //
  //   tuple_replicate   //
  // ------------------- //

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
  tuple_replicate(Arg&& arg)
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
      return std::tuple_cat(std::move(r), tuple_replicate<N - 1>(std::forward<Arg>(arg)));
    }
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_UTILS_HPP
