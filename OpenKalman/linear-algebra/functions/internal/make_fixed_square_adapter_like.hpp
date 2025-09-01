/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/compares_with.hpp"
#include "coordinates/descriptors/Dimensions.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename...Ts>
    using best_desc = std::decay_t<decltype(most_fixed_pattern(std::declval<Ts>()...))>;


    template<typename...Ds, typename Arg, std::size_t...Ix>
    constexpr decltype(auto) make_fixed_square_adapter_like_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      using B = best_desc<Ds..., vector_space_descriptor_of_t<Arg, Ix>...>;
      using F = decltype(make_fixed_size_adapter<std::conditional_t<Ix >= 0, B, B>...>(std::declval<Arg&&>()));
      constexpr bool better = (... or (dynamic_dimension<Arg, Ix> and not dynamic_dimension<F, Ix>));
      if constexpr (better) return F {std::forward<Arg>(arg)};
      else return std::forward<Arg>(arg);
    }
  }


  /**
   * \brief Make the best possible \ref square_shaped, if applicable, derived from the sizes of an object and other info.
   * \tparam Ds Optional vector space descriptors possibly reflecting the square dimension
   * \return (1) A fixed size adapter or (2) a reference to the argument unchanged.
   */
#ifdef __cpp_concepts
  template<coordinates::pattern D = coordinates::Axis, coordinates::pattern...Ds, square_shaped<applicability::permitted> Arg> requires
    (index_count_v<Arg> != dynamic_size) and (... and coordinates::compares_with<D, Ds, equal_to<>, applicability::permitted>)
#else
  template<typename...Ds, typename Arg, std::enable_if_t<
    (... and coordinates::pattern<Ds>) and square_shaped<Arg, applicability::permitted> and
    (index_count_v<Arg> != dynamic_size) and (... and coordinates::compares_with<D, Ds, equal_to<>, applicability::permitted>), int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_square_adapter_like(Arg&& arg)
  {
    return detail::make_fixed_square_adapter_like_impl<Ds...>(std::forward<Arg>(arg), std::make_index_sequence<index_count_v<Arg>>{});
  }


}

#endif
