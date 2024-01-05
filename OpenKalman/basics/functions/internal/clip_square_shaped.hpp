/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref clip_square_shaped function.
 */

#ifndef OPENKALMAN_CLIP_SQUARE_SHAPED_HPP
#define OPENKALMAN_CLIP_SQUARE_SHAPED_HPP


namespace OpenKalman::internal
{

  namespace detail
  {
    template<typename Arg>
    constexpr auto get_smallest_dimension(const Arg& arg) { return std::integral_constant<std::size_t, 1>{}; }


    template<std::size_t I, std::size_t...Is, typename Arg>
    constexpr auto get_smallest_dimension(const Arg& arg)
    {
      auto dim0 = get_index_dimension_of<I>(arg);
      auto dim = get_smallest_dimension<Is...>(arg);
      if constexpr (static_index_value<decltype(dim0)> and static_index_value<decltype(dim0)>)
      {
        if constexpr (dim0 > dim) return dim;
        else return dim0;
      }
      else
      {
        if (dim0 > dim) return static_cast<std::size_t>(dim);
        else return static_cast<std::size_t>(dim0);
      }
    }


    template<std::size_t...Ix, typename Arg>
    constexpr decltype(auto) clip_square_shaped_impl(std::index_sequence<Ix...>, Arg&& arg)
    {
      auto dim = get_smallest_dimension<Ix...>(arg);
      auto ret {make_fixed_size_adapter<decltype(dim)>(get_block(std::forward<Arg>(arg),
        std::forward_as_tuple(std::integral_constant<std::size_t, static_cast<decltype(Ix)>(0)>{}...),
        std::forward_as_tuple((Ix>=0?dim:dim)...)))};
      return ret;
    }

  } // namespace detail


  /**
   * \internal
   * \brief Given inputs to a rank update function, return a writable square matrix
   */
  template<typename Arg>
#ifdef __cpp_concepts
  constexpr square_shaped decltype(auto)
#else
  constexpr decltype(auto)
#endif
  clip_square_shaped(Arg&& arg)
  {
    if constexpr (square_shaped<Arg>) return std::forward<Arg>(arg);
    else return detail::clip_square_shaped_impl(std::make_index_sequence<index_count_v<Arg>>{}, std::forward<Arg>(arg));
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_CLIP_SQUARE_SHAPED_HPP
