/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for make_fixed_square_adapter_like function.
 */

#ifndef OPENKALMAN_MAKE_FIXED_SQUARE_ADAPTER_LIKE_HPP
#define OPENKALMAN_MAKE_FIXED_SQUARE_ADAPTER_LIKE_HPP

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename...Ts>
    using best_desc = std::decay_t<decltype(best_vector_space_descriptor(std::declval<Ts>()...))>;


    template<typename...Ds, typename Arg, std::size_t...Ix>
    constexpr decltype(auto) make_fixed_square_adapter_like_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      using B = best_desc<Ds..., vector_space_descriptor_of_t<Arg, Ix>...>;
      using F = decltype(make_fixed_size_adapter<std::conditional_t<Ix >= 0, B, B>...>(std::declval<Arg&&>()));
      constexpr bool better = (... or (dynamic_dimension<Arg, Ix> and not dynamic_dimension<F, Ix>));
      if constexpr (better) return F {std::forward<Arg>(arg)};
      else return std::forward<Arg>(arg);
    }
  } // namespace detail


  /**
   * \brief Make the best possible \ref square_shaped, if applicable, derived from the sizes of an object and other info.
   * \tparam Ds Optional vector space descriptors possibly reflecting the square dimension
   * \return (1) A fixed size adapter or (2) a reference to the argument unchanged.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor...Ds, square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires
    (index_count_v<Arg> != dynamic_size) and maybe_equivalent_to<Ds...>
#else
  template<typename...Ds, typename Arg, std::enable_if_t<
    (... and vector_space_descriptor<Ds>) and square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
    (index_count_v<Arg> != dynamic_size) and maybe_equivalent_to<Ds...>, int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_square_adapter_like(Arg&& arg)
  {
    return detail::make_fixed_square_adapter_like_impl<Ds...>(std::forward<Arg>(arg), std::make_index_sequence<index_count_v<Arg>>{});
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAKE_FIXED_SQUARE_ADAPTER_LIKE_HPP
