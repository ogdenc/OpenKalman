/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref patterns::to_extents.
 */

#ifndef OPENKALMAN_TO_EXTENTS_HPP
#define OPENKALMAN_TO_EXTENTS_HPP

#include "patterns/functions/get_dimension.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/traits/pattern_collection_element.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
    template<typename P, std::size_t curr_rank = collections::size_of_v<P>>
    struct derive_rank
      : std::integral_constant<
        std::size_t,
        dimension_of_v<pattern_collection_element_t<curr_rank - 1, P>> == 1 ?
          derive_rank<P, curr_rank - 1>::value :
          curr_rank>
    {};

    template<typename P>
    struct derive_rank<P, 0> : std::integral_constant<std::size_t, 0> {};


    template<std::size_t previous_i, std::size_t...SDs, typename P, typename...Ds>
    constexpr auto
    derive_extents(const P& p, Ds...ds)
    {
      if constexpr (previous_i == 0)
      {
        return stdex::extents<std::size_t, SDs...>{std::move(ds)...};
      }
      else
      {
        constexpr std::size_t i = previous_i - 1;
        auto d = patterns::get_dimension(get_pattern<i>(p));
        if constexpr (values::fixed<decltype(d)>)
          return derive_extents<i, values::fixed_value_of_v<decltype(d)>, SDs...>(p, std::move(ds)...);
        else
          return derive_extents<i, stdex::dynamic_extent, SDs...>(p, std::move(d), std::move(ds)...);
      }
    }

  }


  /**
   * \brief Convert a \ref pattern_collection to std::extents.
   * \tparam rank The rank of the result. The rank must be large enough to hold all non-unitary extents.
   * If necessary, the template parameters of the result
   * will be padded with sufficient trailing 1 values to meet this rank.
   */
#ifdef __cpp_lib_constexpr_vector
  template<std::size_t rank, pattern_collection P> requires
    values::fixed_value_compares_with<collections::size_of<P>, stdex::dynamic_extent, &stdex::is_neq>
#else
  template<std::size_t rank, typename P, std::enable_if_t<
    pattern_collection<P> and
    values::fixed_value_compares_with<collections::size_of<P>, stdex::dynamic_extent, &stdex::is_neq>, int> = 0>
#endif
  constexpr auto
  to_extents(P&& p)
  {
    static_assert(rank >= detail::derive_rank<P>::value, "Rank must be large enough to hold all non-unitary extents.");
    return detail::derive_extents<rank>(p);
  }


  /**
   * \overload
   */
  template<std::size_t rank, typename IndexType, std::size_t...Extents>
  constexpr decltype(auto)
  to_extents(const stdex::extents<IndexType, Extents...>& p)
  {
    constexpr std::size_t N = detail::derive_rank<stdex::extents<IndexType, Extents...>>::value;
    static_assert(rank >= N);
    if constexpr (rank > N)
      return detail::derive_extents<rank>(p);
    else
      return p;
  }


  /**
   * \overload
   */
  template<std::size_t rank, typename IndexType, std::size_t...Extents>
  constexpr auto
  to_extents(stdex::extents<IndexType, Extents...>&& p)
  {
    constexpr std::size_t N = detail::derive_rank<stdex::extents<IndexType, Extents...>>::value;
    static_assert(rank >= N);
    if constexpr (rank > N)
      return detail::derive_extents<rank>(std::move(p));
    else
      return std::move(p);
  }


  /**
   * \overload
   * \brief Derive the rank from the argument. The number of extents will be minimized.
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern_collection P> requires
    values::fixed_value_compares_with<collections::size_of<P>, stdex::dynamic_extent, &stdex::is_neq>
#else
  template<typename P, std::enable_if_t<
    pattern_collection<P> and
    values::fixed_value_compares_with<collections::size_of<P>, stdex::dynamic_extent, &stdex::is_neq>, int> = 0>
#endif
  constexpr auto
  to_extents(P&& p)
  {
    return detail::derive_extents<detail::derive_rank<P>::value>(p);
  }


}


#endif
