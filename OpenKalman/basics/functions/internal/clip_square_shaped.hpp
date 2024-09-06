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
    template<std::size_t...Ix, typename Arg>
    constexpr decltype(auto) clip_square_shaped_impl(std::index_sequence<Ix...>, Arg&& arg)
    {
      auto dim = internal::smallest_vector_space_descriptor(get_index_dimension_of<0>(arg), get_index_dimension_of<1>(arg));
      auto ret {get_slice(std::forward<Arg>(arg),
        std::forward_as_tuple(std::integral_constant<std::size_t, static_cast<decltype(Ix)>(0)>{}...),
        std::forward_as_tuple((Ix==0?dim:dim)...))};
      return ret;
    }

  } // namespace detail


  /**
   * \internal
   * \brief Given inputs to a rank update function, return a writable square matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr square_shaped<Qualification::depends_on_dynamic_shape> decltype(auto)
#else
  template<typename Arg>
  constexpr decltype(auto)
#endif
  clip_square_shaped(Arg&& arg)
  {
    if constexpr (square_shaped<Arg>) return std::forward<Arg>(arg);
    else
    {
      static_assert(index_count_v<Arg> <= 2);
      return detail::clip_square_shaped_impl(std::make_index_sequence<index_count_v<Arg>>{}, std::forward<Arg>(arg));
    }
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_CLIP_SQUARE_SHAPED_HPP
