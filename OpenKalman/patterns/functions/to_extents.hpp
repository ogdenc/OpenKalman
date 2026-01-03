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

namespace OpenKalman::patterns
{
  namespace detail
  {
    template<std::size_t N, std::size_t i = 0, std::size_t...SDs, typename P, typename...Ds>
    constexpr auto
    derive_extents(const P& p, Ds...ds)
    {
      if constexpr (i < N)
      {
        auto d = patterns::get_dimension(collections::get<i>(p));
        if constexpr (values::fixed<decltype(d)>)
          return derive_extents<N, i + 1, SDs..., values::fixed_value_of_v<decltype(d)>>(p, std::move(ds)...);
        else
          return derive_extents<N, i + 1, SDs..., stdex::dynamic_extent>(p, std::move(ds)..., std::move(d));
      }
      else return stdex::extents<std::size_t, SDs...>{std::move(ds)...};
    }
  }


  /**
   * \brief Convert a \ref pattern_collection to std::extents.
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern_collection P>
#else
  template<typename P, std::enable_if_t<pattern_collection<P>, int> = 0>
#endif
  constexpr auto
  to_extents(P&& p)
  {
    return detail::derive_extents<collections::size_of_v<P>>(p);
  }


  /**
   * \overload
   */
  template<typename IndexType, std::size_t...Extents>
  constexpr decltype(auto)
  to_extents(const stdex::extents<IndexType, Extents...>& e)
  {
    return e;
  }


  /**
   * \overload
   */
  template<typename IndexType, std::size_t...Extents>
  constexpr auto
  to_extents(stdex::extents<IndexType, Extents...>&& e)
  {
    return std::move(e);
  }


}


#endif
