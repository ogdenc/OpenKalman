/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for make_fixed_size_adapter_like function.
 */

#ifndef OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_LIKE_HPP
#define OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_LIKE_HPP

namespace OpenKalman::internal
{
  namespace detail
  {
    template<std::size_t I, typename...Ts>
    constexpr decltype(auto) best_desc_Ts_impl(const Ts&...ts)
    {
      return most_fixed_pattern(get_pattern_collection<I>(ts)...);
    }


    template<typename...Ts, typename Arg, std::size_t...Ix>
    constexpr decltype(auto) make_fixed_size_adapter_like_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      if constexpr (sizeof...(Ts) == 0) return std::forward<Arg>(arg);
      else
      {
        using F = decltype(make_fixed_size_adapter<decltype(best_desc_Ts_impl<Ix>(std::declval<Ts>()...))...>(std::declval<Arg&&>()));
        constexpr bool better = (... or (dynamic_dimension<Arg, Ix> and not dynamic_dimension<F, Ix>));
        if constexpr (better) return F {std::forward<Arg>(arg)};
        else return std::forward<Arg>(arg);
      }
    }
  }


  /**
   * \brief Make the best possible \ref FixedSizeAdapter, if applicable, derived from the sizes of several objects.
   * \tparam Ts Optional indexible objects on which to base the fixed dimensions
   * \return (1) A fixed size adapter or (2) a reference to the argument unchanged.
   */
#ifdef __cpp_concepts
  template<indexible...Ts, vector_space_descriptors_may_match_with<Ts...> Arg> requires (index_count_v<Arg> != dynamic_size)
#else
  template<typename...Ts, typename Arg, std::enable_if_t<(... and indexible<Ts>) and
    vector_space_descriptors_may_match_with<Arg, Ts...> and (index_count_v<Arg> != dynamic_size), int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter_like(Arg&& arg)
  {
    constexpr auto min_count = std::min({index_count_v<Arg>,
      (index_count_v<Ts> == dynamic_size ? index_count_v<Arg> : index_count_v<Ts>)...});
    return detail::make_fixed_size_adapter_like_impl<Ts...>(std::forward<Arg>(arg), std::make_index_sequence<min_count>{});
  }


}

#endif
