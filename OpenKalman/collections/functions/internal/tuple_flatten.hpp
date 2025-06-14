/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref collections::tuple_flatten.
 */

#ifndef OPENKALMAN_TUPLE_FLATTEN_HPP
#define OPENKALMAN_TUPLE_FLATTEN_HPP

#include <type_traits>
#include <tuple>
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename Arg> constexpr auto
    tuple_flatten_impl(Arg&&); // forward declaration


    template<typename Arg, std::size_t...Ix>
    constexpr auto
    tuple_flatten_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      if constexpr ((... or tuple_like<std::tuple_element_t<Ix, std::decay_t<Arg>>>))
        return std::tuple_cat(tuple_flatten_impl(std::get<Ix>(std::forward<Arg>(arg)))...);
      else
        return std::forward<Arg>(arg);
    }


    template<typename Arg>
    constexpr auto
    tuple_flatten_impl(Arg&& arg)
    {
      if constexpr (tuple_like<Arg>)
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
  template<tuple_like Arg>
  constexpr tuple_like auto
#else
  template<typename Arg, std::enable_if_t<tuple_like<Arg>, int> = 0>
  constexpr auto
#endif
  tuple_flatten(Arg&& arg)
  {
    return detail::tuple_flatten_impl(std::forward<Arg>(arg));
  }

} // namespace OpenKalman::collections

#endif //OPENKALMAN_TUPLE_FLATTEN_HPP
