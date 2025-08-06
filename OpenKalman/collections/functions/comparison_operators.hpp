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
 * \internal
 * \brief Definition of \ref collections::compare and comparison operations for \ref collections::collection "collections"
 */

#ifndef OPENKALMAN_COLLECTION_COMPARE_HPP
#define OPENKALMAN_COLLECTION_COMPARE_HPP

#include <type_traits>
#include "basics/basics.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
  namespace internal
  {
    namespace detail
    {
      template<std::size_t i = 0, typename T1, typename T2>
      constexpr auto
      fixed_compare(const T1& lhs, const T2& rhs)
      {
        using namespace std;
        constexpr auto ix = std::integral_constant<std::size_t, i>{};
        if constexpr (i == size_of_v<T1> and i == size_of_v<T2>) { return stdcompat::partial_ordering::equivalent; }
        else if constexpr (i == size_of_v<T1>) { return stdcompat::partial_ordering::less; }
        else if constexpr (i == size_of_v<T2>) { return stdcompat::partial_ordering::greater; }
        else if (collections::get(lhs, ix) == collections::get(rhs, ix)) { return fixed_compare<i + 1>(lhs, rhs); }
        else
        {
          auto cmp = stdcompat::compare_three_way{}(collections::get(lhs, ix), collections::get(rhs, ix));
          if (cmp == 0) return stdcompat::partial_ordering::equivalent;
          if (cmp < 0) return stdcompat::partial_ordering::less;
          if (cmp > 0) return stdcompat::partial_ordering::greater;
          return stdcompat::partial_ordering::unordered;
        }
      }
    }


    /**
     * \internal
     * \brief A callable object that compares two \ref collections::collections "collections".
     * \details This is the default case, but the class can be specialized for particular types of collections.
     */
    template<typename Lhs, typename Rhs>
    struct compare
    {
      constexpr stdcompat::partial_ordering
      operator() (const Lhs& lhs, const Rhs& rhs) const
      {
#ifdef __cpp_lib_ranges
        namespace ranges = std::ranges;
#endif
        if constexpr (size_of_v<Lhs> != dynamic_size and size_of_v<Rhs> != dynamic_size)
        {
          return detail::fixed_compare(lhs, rhs);
        }
        else
        {
          return stdcompat::lexicographical_compare_three_way(
            stdcompat::ranges::begin(lhs),
            stdcompat::ranges::end(lhs),
            stdcompat::ranges::begin(rhs),
            stdcompat::ranges::end(rhs));
        }
      }
    };

  }


#ifdef __cpp_impl_three_way_comparison
  /**
   * \brief Compare two \ref collections::collections "collections", at least one of which is a \ref collection_view.
   */
  template<collection Lhs, collection Rhs> requires collection_view<Lhs> or collection_view<Rhs>
  constexpr std::partial_ordering
  operator<=>(const Lhs& lhs, const Rhs& rhs) noexcept
  {
    return internal::compare<Lhs, Rhs>{}(lhs, rhs);
  }


  /**
   * \brief Compare two \ref collections::collections "collections" for equality, at least one of which is a \ref collection_view.
   */
  template<collection Lhs, collection Rhs> requires collection_view<Lhs> or collection_view<Rhs>
  constexpr bool
  operator==(const Lhs& lhs, const Rhs& rhs) noexcept
  {
    return stdcompat::is_eq(operator<=>(lhs, rhs));
  }
#else
  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>), int> = 0>
  constexpr bool operator==(const Lhs& lhs, const Rhs& rhs) { return internal::compare<Lhs, Rhs>{}(lhs, rhs) == 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>), int> = 0>
  constexpr bool operator!=(const Lhs& lhs, const Rhs& rhs) { return not (internal::compare<Lhs, Rhs>{}(lhs, rhs) == 0); }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>), int> = 0>
  constexpr bool operator<(const Lhs& lhs, const Rhs& rhs) { return internal::compare<Lhs, Rhs>{}(lhs, rhs) < 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>), int> = 0>
  constexpr bool operator>(const Lhs& lhs, const Rhs& rhs) { return internal::compare<Lhs, Rhs>{}(lhs, rhs) > 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>), int> = 0>
  constexpr bool operator<=(const Lhs& lhs, const Rhs& rhs) { return internal::compare<Lhs, Rhs>{}(lhs, rhs) <= 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (collection_view<Lhs> or collection_view<Rhs>), int> = 0>
  constexpr bool operator>=(const Lhs& lhs, const Rhs& rhs) { return internal::compare<Lhs, Rhs>{}(lhs, rhs) >= 0; }
#endif

}

#endif //OPENKALMAN_COLLECTION_COMPARE_HPP
