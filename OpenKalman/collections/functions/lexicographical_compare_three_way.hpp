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
 * \file
 * \brief Definition of \ref lexicographical_compare_three_way for \ref collections::collection "collections"
 */

#ifndef OPENKALMAN_COLLECTION_LEXICOGRAPHICAL_COMPARE_THREE_WAY_HPP
#define OPENKALMAN_COLLECTION_LEXICOGRAPHICAL_COMPARE_THREE_WAY_HPP

#include <type_traits>
#include "basics/basics.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/collection_view.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<std::size_t i = 0, typename Lhs, typename Rhs>
    constexpr stdex::partial_ordering
    fixed_lexicographical_compare(const Lhs& lhs, const Rhs& rhs)
    {
      using ix = std::integral_constant<std::size_t, i>;
      constexpr bool exhaust_lhs = values::size_compares_with<size_of<Lhs>, ix>;
      constexpr bool exhaust_rhs = values::size_compares_with<size_of<Rhs>, ix>;
      constexpr bool dyn_lhs = values::fixed_value_compares_with<size_of<Lhs>, stdex::dynamic_extent>;
      constexpr bool dyn_rhs = values::fixed_value_compares_with<size_of<Rhs>, stdex::dynamic_extent>;

      if constexpr (exhaust_lhs and exhaust_rhs)
      {
        return stdex::partial_ordering::equivalent;
      }
      else if constexpr (exhaust_lhs)
      {
        if constexpr (dyn_rhs) if (i == get_size(rhs)) return stdex::partial_ordering::equivalent;
        return stdex::partial_ordering::less;
      }
      else if constexpr (exhaust_rhs)
      {
        if constexpr (dyn_lhs) if (i == get_size(lhs)) return stdex::partial_ordering::equivalent;
        return stdex::partial_ordering::greater;
      }
      else // if constexpr (not exhaust_lhs and not exhaust_rhs)
      {
        if constexpr (dyn_lhs) if (i == get_size(lhs)) return stdex::partial_ordering::less;
        if constexpr (dyn_rhs) if (i == get_size(rhs)) return stdex::partial_ordering::greater;
        if (auto cmp = stdex::compare_three_way{}(collections::get<i>(lhs), collections::get<i>(rhs));
            stdex::is_eq(cmp))
          return fixed_lexicographical_compare<i + 1>(lhs, rhs);
        else return cmp;
      }
    }
  }


  /**
   * \brief Compares two \ref collections::collection "collections".
   * \details This is the default case in which at least one of the arguments is a \ref collection_view.
   */
#ifdef __cpp_concepts
  template<collection Lhs, collection Rhs> requires
    (collection_view<Lhs> or collection_view<Rhs>) and
    (sized<Lhs> or sized<Rhs>)
#else
  template<typename Lhs, typename Rhs, std::enable_if_t<
    collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>) and
    (sized<Lhs> or sized<Rhs>), int> = 0>
#endif
  constexpr stdex::partial_ordering
  lexicographical_compare_three_way(const Lhs& lhs, const Rhs& rhs)
  {
    if constexpr (values::fixed_value_compares_with<size_of<Lhs>, stdex::dynamic_extent, &stdex::is_neq> or
      values::fixed_value_compares_with<size_of<Rhs>, stdex::dynamic_extent, &stdex::is_neq>)
    {
      return detail::fixed_lexicographical_compare(lhs, rhs);
    }
    else if constexpr (stdex::ranges::common_range<Lhs> and stdex::ranges::common_range<Rhs>)
    {
      return stdex::lexicographical_compare_three_way(
        stdex::ranges::begin(lhs),
        stdex::ranges::end(lhs),
        stdex::ranges::begin(rhs),
        stdex::ranges::end(rhs));
    }
    else
    {
      auto f1 = stdex::ranges::begin(lhs);
      const auto l1 = stdex::ranges::end(lhs);
      auto f2 = stdex::ranges::begin(rhs);
      const auto l2 = stdex::ranges::end(rhs);
      bool exhaust1 = (f1 == l1);
      bool exhaust2 = (f2 == l2);
      for (; not exhaust1 and not exhaust2; exhaust1 = (++f1 == l1), exhaust2 = (++f2 == l2))
        if (auto c = stdex::compare_three_way{}(*f1, *f2); stdex::is_neq(c)) return c;
      return !exhaust1 ? stdex::partial_ordering::greater:
             !exhaust2 ? stdex::partial_ordering::less:
                         stdex::partial_ordering::equivalent;
    }
  }


}

#endif
