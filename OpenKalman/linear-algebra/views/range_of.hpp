/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref views::range_of.
 */

#ifndef OPENKALMAN_RANGE_OF_HPP
#define OPENKALMAN_RANGE_OF_HPP

#include "collections/collections.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/index_collection_for.hpp"
#include "linear-algebra/traits/get_index_extent.hpp"
#include "linear-algebra/traits/access.hpp"
#include "linear-algebra/concepts/empty_object.hpp"

namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct range_of_adaptor
    {
#ifdef __cpp_concepts
      template<indexible T, collections::index Indices> requires
        index_collection_for<decltype(views::concat(std::tuple{0_uz}, std::declval<Indices>())), T>
      constexpr collections::collection auto
#else
      template<typename T, typename Indices, std::enable_if_t<indexible<T> and collections::index<Indices> and
        index_collection_for<decltype(views::concat(std::tuple{0_uz}, std::declval<Indices>())), T>, int> = 0>
      constexpr auto
#endif
      operator() (T&& t, Indices indices) const
      {
        return collections::views::generate(
          [m = std::tuple<T&&>{std::forward<T>(t)}, ind = std::move(indices)](auto i){
            return access(std::get<0>(m), views::concat(std::tuple{i}, ind));
            }, get_index_extent<0>(t));
      }


#ifdef __cpp_concepts
      template<indexible T, values::index...I> requires
        (not empty_object<T>) and
        values::size_compares_with<std::integral_constant<std::size_t, sizeof...(I) + 1>, index_count<T>, &stdex::is_gteq>
      constexpr collections::collection auto
#else
      template<typename T, typename...I, std::enable_if_t<indexible<T> and (... and values::index<I>) and
        (not empty_object<Arg>) and
        values::size_compares_with<std::integral_constant<std::size_t, sizeof...(I) + 1>, index_count<T>, &stdex::is_gteq>, int> = 0>
      constexpr auto
#endif
      operator() (T&& t, I...i) const
      {
        return operator()(std::forward<T>(t), std::array<std::size_t, sizeof...(I)>{std::move(i)...});
      }

    };

  }


  /**
   * \brief A RangeAdapterObject that creates a 1D range based on a slice of an \ref indexible object.
   */
  inline constexpr detail::range_of_adaptor range_of;

}

#endif
