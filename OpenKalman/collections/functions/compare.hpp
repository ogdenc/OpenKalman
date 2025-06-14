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
#include <tuple>
#include "basics/compatibility/language-features.hpp"
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "collections/concepts/collection_view.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
#ifdef __cpp_lib_ranges
    template<typename T>
    concept comparable = collection_view<T> and (size_of_v<T> != dynamic_size) or std::ranges::input_range<T>;
#else
    template<typename T>
    inline constexpr bool comparable = collection_view<T> and (size_of_v<T> != dynamic_size) or ranges::input_range<T>;
#endif


    template<std::size_t i = 0, typename T1, typename T2>
    constexpr auto
    fixed_compare(const T1& lhs, const T2& rhs)
    {
      using namespace std;
      constexpr auto ix = std::integral_constant<std::size_t, i>{};
      if constexpr (i == size_of_v<T1> and i == size_of_v<T2>) { return partial_ordering::equivalent; }
      else if constexpr (i == size_of_v<T1>) { return partial_ordering::less; }
      else if constexpr (i == size_of_v<T2>) { return partial_ordering::greater; }
      else if (get(lhs, ix) == get(rhs, ix)) { return fixed_compare<i + 1>(lhs, rhs); }
      else
      {
        auto cmp = compare_three_way{}(get(lhs, ix), get(rhs, ix));
        if (cmp == 0) return partial_ordering::equivalent;
        if (cmp < 0) return partial_ordering::less;
        if (cmp > 0) return partial_ordering::greater;
        return partial_ordering::unordered;
      }
    }
  }


  /**
   * \brief Compare two \ref collections::collections "collections"
   */
#ifdef __cpp_impl_three_way_comparison
  template<detail::comparable Lhs, detail::comparable Rhs>
  constexpr std::partial_ordering
  compare(const Lhs& lhs, const Rhs& rhs)
  {
#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif
    if constexpr (size_of_v<Lhs> != dynamic_size and size_of_v<Rhs> != dynamic_size) { return detail::fixed_compare(lhs, rhs); }
    else { return std::lexicographical_compare_three_way(ranges::begin(lhs), ranges::end(lhs), ranges::begin(rhs), ranges::end(rhs)); }
  }


  /**
   * \brief Compare two \ref collections::collections "collections"
   */
  template<detail::comparable Lhs, detail::comparable Rhs>
  constexpr std::partial_ordering
  operator<=>(const Lhs& lhs, const Rhs& rhs) noexcept
  {
    return compare(lhs, rhs);
  }


  /**
   * \brief Compare two \ref collections::collections "collections" for equality
   */
  template<detail::comparable Lhs, detail::comparable Rhs>
  constexpr bool
  operator==(const Lhs& lhs, const Rhs& rhs) noexcept
  {
    return std::is_eq(operator<=>(lhs, rhs));
  }
#else
  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr partial_ordering
  compare(const Lhs& lhs, const Rhs& rhs)
  {
    if constexpr (size_of_v<Lhs> != dynamic_size and size_of_v<Rhs> != dynamic_size) return detail::fixed_compare(lhs, rhs);
    else
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      auto l = ranges::begin(lhs);
      auto r = ranges::begin(rhs);
      auto le = ranges::end(lhs);
      auto re = ranges::end(rhs);
      for (; l != le and r != re; ++l, ++r)
      {
        if (*l == *r) { continue; }
        if (*l < *r) { return partial_ordering::less; }
        if (*l > *r) { return partial_ordering::greater; }
        return partial_ordering::unordered;
      }
      if (l == le and r == re) { return partial_ordering::equivalent; }
      if (l == le) { return partial_ordering::less; }
      if (r == re) { return partial_ordering::greater; }
      return partial_ordering::unordered;
    }
  }


  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr bool operator==(const Lhs& lhs, const Rhs& rhs) { return compare(lhs, rhs) == 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr bool operator!=(const Lhs& lhs, const Rhs& rhs) { return not (compare(lhs, rhs) == 0); }

  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr bool operator<(const Lhs& lhs, const Rhs& rhs) { return compare(lhs, rhs) < 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr bool operator>(const Lhs& lhs, const Rhs& rhs) { return compare(lhs, rhs) > 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr bool operator<=(const Lhs& lhs, const Rhs& rhs) { return compare(lhs, rhs) <= 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<detail::comparable<Lhs> and detail::comparable<Rhs>, int> = 0>
  constexpr bool operator>=(const Lhs& lhs, const Rhs& rhs) { return compare(lhs, rhs) >= 0; }
#endif

}

#endif //OPENKALMAN_COLLECTION_COMPARE_HPP
