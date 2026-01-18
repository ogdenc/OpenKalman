/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref collections::views::all and \ref collections::views::all_t.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_ALL_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_ALL_HPP

#include "basics/basics.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "from_tuple_like.hpp"
#include "from_range.hpp"
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct all_closure : stdex::ranges::range_adaptor_closure<all_closure>
    {
#ifdef __cpp_concepts
      template<viewable_collection R>
      constexpr collection_view auto
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
      constexpr auto
#endif
      operator() (R&& r) const
      {
        if constexpr (collection_view<R> and (collections::tuple_like<R> or
          (uniformly_gettable<R> and not values::fixed_value_compares_with<size_of<R>, stdex::dynamic_extent, &stdex::is_neq>)))
        {
          return std::forward<R>(r);
        }
        else if constexpr (stdex::ranges::random_access_range<R> and stdex::ranges::viewable_range<R>)
        {
          return from_range {std::forward<R>(r)};
        }
        else //if constexpr (viewable_tuple_like<R>)
        {
          return from_tuple_like {std::forward<R>(r)};
        }
      }
    };
  }


  /**
   * \brief a std::ranges::range_adaptor_closure which returns a view to all members of its \ref collection argument.
   * \details The result will be a \ref collection_view and, if it has a fixed size, will be \ref tuple_like
   * Examples:
   * \code
   * static_assert(equal_to{}(views::all{std::tuple{4, 5.}}, std::tuple{4, 5.}));
   * static_assert(equal_to{}(std::tuple{4, 5.}, views::all(std::tuple{4, 5.})));
   * static_assert(equal_to{}(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 6}));
   * static_assert(equal_to{}(std::array{4, 5, 6}, views::all(std::array{4, 5, 6})));
   * \endcode
   */
  inline constexpr detail::all_closure all;


  /**
   * \brief Calculates the suitable \ref collection_view type of a \ref viewable_collection type.
   * \details This is rougly equivalent to std::ranges::views::all, except in the context of a \ref collection
   */
#ifdef __cpp_concepts
  template<viewable_collection R>
#else
  template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
  using all_t = decltype(all(std::declval<R>()));

}


#endif
