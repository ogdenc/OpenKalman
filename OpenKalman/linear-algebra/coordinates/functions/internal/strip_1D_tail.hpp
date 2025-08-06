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
 * \internal
 * \file
 * \brief Definition for \ref strip_1D_tail.
 */

#ifndef OPENKALMAN_STRIP_1D_TAIL_HPP
#define OPENKALMAN_STRIP_1D_TAIL_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"

namespace OpenKalman::coordinates::internal
{
  namespace detail
  {
    template<typename D, typename I>
    constexpr decltype(auto)
    strip_1D_tail_dynamic(D&& d, I i)
    {
      if constexpr (values::fixed<I>)
      {
        if constexpr (values::fixed_number_of_v<I> > 0_uz)
        {
          auto new_i = values::operation(std::minus{}, i, std::integral_constant<std::size_t, 1>{});
          if (stdcompat::is_eq(compare(collections::get(d, new_i), Dimensions<1>{})))
            return strip_1D_tail_dynamic(std::forward<D>(d), new_i);
        }
        return collections::views::slice(std::forward<D>(d), std::integral_constant<std::size_t, 0>{}, values::to_number(i));
      }
      else
      {
        decltype(auto) d_coll = collections::views::all(std::forward<D>(d));
        while (i > 0_uz)
        {
          if (stdcompat::is_neq(compare(collections::get(d_coll, --i), Dimensions<1>{}))) { ++i; break; }
        }
        return collections::views::slice(std::forward<decltype(d_coll)>(d_coll), std::integral_constant<std::size_t, 0>{}, i);
      }
    }


    template<std::size_t i, typename D>
    constexpr decltype(auto)
    strip_1D_tail_fixed(D&& d)
    {
      if constexpr (i == 0)
      {
        return stdcompat::ranges::views::empty<Dimensions<1>>;
      }
      else
      {
        using LastElem = decltype(collections::get(d, std::integral_constant<std::size_t, i - 1_uz>{}));
        if constexpr (compares_with<LastElem, Dimensions<1>>)
        {
          return strip_1D_tail_fixed<i - 1_uz>(std::forward<D>(d));
        }
        else if constexpr (compares_with<LastElem, Dimensions<1>, &stdcompat::is_eq, Applicability::permitted>)
        {
          return strip_1D_tail_dynamic(std::forward<D>(d), std::integral_constant<std::size_t, i>{});
        }
        else
        {
          return collections::views::slice(std::forward<D>(d),
            std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, i>{});
        }
      }
    }
  }


  /**
   * \internal
   * \brief Remove any trailing, one-dimensional \ref pattern objects from a \ref pattern_collection.
   * \return A \ref pattern_collection containing a potentially shortened collection of \ref pattern objects
   */
#ifdef __cpp_concepts
  template<pattern_collection P> requires collections::sized<P>
  constexpr pattern_collection decltype(auto)
#else
  template<typename P, std::enable_if_t<pattern_collection<P> and collections::sized<P>, int> = 0>
  constexpr decltype(auto)
#endif
  strip_1D_tail(P&& p)
  {
    if constexpr (collections::size_of_v<P> == dynamic_size)
      return detail::strip_1D_tail_dynamic(std::forward<P>(p), collections::get_size(p));
    else
      return detail::strip_1D_tail_fixed<collections::size_of_v<P>>(std::forward<P>(p));
  }


}


#endif
