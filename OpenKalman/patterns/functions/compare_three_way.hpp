/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref patterns::compare_three_way.
 */

#ifndef OPENKALMAN_PATTERNS_COMPARE_THREE_WAY_HPP
#define OPENKALMAN_PATTERNS_COMPARE_THREE_WAY_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/functions/get_dimension.hpp"
#include "patterns/functions/internal/get_descriptor_hash_code.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
    template<std::size_t ia = 0, std::size_t ib = 0, typename A, typename B, typename C>
    constexpr auto
    compare_three_way_fixed(const A& a, const B& b, const C& c, std::size_t abank = 0, std::size_t bbank = 0)
    {
      if constexpr (ia < collections::size_of_v<A>)
      {
        auto ai = collections::get<ia>(a);
        if constexpr (ib < collections::size_of_v<B>)
        {
          auto bi = collections::get<ib>(b);
          constexpr bool ae = euclidean_pattern<decltype(ai)>;
          constexpr bool be = euclidean_pattern<decltype(bi)>;
          if constexpr (ae and be)
            return compare_three_way_fixed<ia + 1, ib + 1>(a, b, c, abank + get_dimension(ai), bbank + get_dimension(bi));
          else if constexpr (ae)
            return compare_three_way_fixed<ia + 1, ib>(a, b, c, abank + get_dimension(ai), bbank);
          else if constexpr (be)
            return compare_three_way_fixed<ia, ib + 1>(a, b, c, abank, bbank + get_dimension(bi));
          else
          {
            if (abank == bbank)
              if (internal::get_descriptor_hash_code(ai) == internal::get_descriptor_hash_code(bi))
                return values::cast_to<stdex::partial_ordering>(compare_three_way_fixed<ia + 1, ib + 1>(a, b, c));
            return stdex::partial_ordering::unordered;
          }
        }
        else if constexpr (euclidean_pattern<decltype(ai)>)
          return compare_three_way_fixed<ia + 1, ib>(a, b, c, abank + get_dimension(ai), bbank);
        else if (abank >= bbank)
          return stdex::partial_ordering::greater;
        else
          return stdex::partial_ordering::unordered;
      }
      else if constexpr (ib < collections::size_of_v<B>) // ia >= collections::size_of_v<A>
      {
        auto bi = collections::get<ib>(b);
        if constexpr (euclidean_pattern<decltype(bi)>) return compare_three_way_fixed<ia, ib + 1>(a, b, c, abank, bbank + get_dimension(bi));
        else if (abank <= bbank) return stdex::partial_ordering::less;
        else return stdex::partial_ordering::unordered;
      }
      else
      {
        return stdex::invoke(c, abank, bbank);
      }
    }


    template<typename A, typename B, typename C, typename Ia = std::integral_constant<std::size_t, 0>, typename Ib = std::integral_constant<std::size_t, 0>>
    constexpr stdex::partial_ordering
    compare_three_way_impl(const A& a, const B& b, const C& c, Ia ia = {}, Ib ib = {}, std::size_t abank = 0, std::size_t bbank = 0)
    {
      if (ia < collections::get_size(a))
      {
        auto a_i = collections::get_element(a, ia);
        auto ae = get_is_euclidean(a_i);
        if (ib < collections::get_size(b))
        {
          auto b_i = collections::get_element(b, ib);
          auto be = get_is_euclidean(b_i);
          if (ae or be)
            return compare_three_way_impl(a, b, c,
              ae ? ia + 1_uz : ia,
              be ? ib + 1_uz : ib,
              ae ? abank + get_dimension(a_i) : abank,
              be ? bbank + get_dimension(b_i) : bbank);
          else if (internal::get_descriptor_hash_code(a_i) == internal::get_descriptor_hash_code(b_i) and abank == bbank)
            return compare_three_way_impl(a, b, c, ia + 1_uz, ib + 1_uz);
          else
            return stdex::partial_ordering::unordered;
        }
        else
        {
          if (ae)
            return compare_three_way_impl(a, b, c, ia + 1_uz, ib, abank + get_dimension(a_i), bbank);
          else if (abank >= bbank)
            return stdex::partial_ordering::greater;
          else
            return stdex::partial_ordering::unordered;
        }
      }
      else if (ib < collections::get_size(b))
      {
        auto b_i = collections::get_element(b, ib);
        if (get_is_euclidean(b_i))
          return compare_three_way_impl(a, b, c, ia, ib + 1_uz, abank, bbank + get_dimension(b_i));
        else if (abank <= bbank)
          return stdex::partial_ordering::less;
        else
          return stdex::partial_ordering::unordered;
      }
      else return stdex::invoke(c, abank, bbank);
    }

  }


  /**
   * \brief Compare two \ref patterns::pattern objects lexicographically.
   * \detail Consecutive \ref euclidean_pattern arguments are consolidated before the comparison occurs.
   * \tparam Comparison A callable comparison function compatible with std::partial_ordering, such as std::compare_three_way
   */
#ifdef __cpp_concepts
  template<pattern A, pattern B, typename Comparison = stdex::compare_three_way>
  requires std::is_invocable_r_v<stdex::partial_ordering, Comparison, std::size_t, std::size_t>
  constexpr std::convertible_to<stdex::partial_ordering> auto
#else
  template<typename A, typename B, typename Comparison = stdex::compare_three_way,
    std::enable_if_t<pattern<A> and pattern<B> and
      std::is_invocable_r<stdex::partial_ordering, Comparison, std::size_t, std::size_t>::value, int> = 0>
  constexpr auto
#endif
  compare_three_way(A&& a, B&& b, const Comparison& c = {})
  {
    if constexpr (euclidean_pattern<A> and euclidean_pattern<B> and
      (descriptor<A> or collections::sized<A>) and (descriptor<B> or collections::sized<B>))
    {
      return values::cast_to<stdex::partial_ordering>(values::operation(c, get_dimension(a), get_dimension(b)));
    }

    else if constexpr (descriptor<A> and descriptor<B>)
    {
      if (get_is_euclidean(a) and get_is_euclidean(b))
        return static_cast<stdex::partial_ordering>(stdex::invoke(c, values::to_value_type(get_dimension(a)), values::to_value_type(get_dimension(b))));
      else if (stdex::is_eq(stdex::invoke(c, internal::get_descriptor_hash_code(a), internal::get_descriptor_hash_code(b))))
        return stdex::partial_ordering::equivalent;
      else
        return stdex::partial_ordering::unordered;
    }
    else if constexpr (descriptor<A>)
    {
      return compare_three_way(stdex::ranges::views::single(stdex::cref(a)), b, c);
    }
    else if constexpr (descriptor<B>)
    {
      return compare_three_way(a, stdex::ranges::views::single(stdex::cref(b)), c);
    }

    else if constexpr (not collections::sized<A> and not collections::sized<B>)
    {
      using RA = stdex::ranges::range_value_t<A>;
      using RB = stdex::ranges::range_value_t<B>;
      // The situation where both RA and RB are zero at compile time is already handled in the euclidean case above
      if constexpr (values::fixed_value_compares_with<dimension_of<RA>, 0>)
        return compare_three_way(std::array<Dimensions<0>, 0>{}, b, c);
      else if constexpr (values::fixed_value_compares_with<dimension_of<RB>, 0>)
        return compare_three_way(a, std::array<Dimensions<0>, 0>{}, c);
      else if constexpr (values::fixed<decltype(internal::get_descriptor_hash_code(std::declval<RA>()))> and
        values::fixed<decltype(internal::get_descriptor_hash_code(std::declval<RB>()))>)
      {
        auto cmp = values::cast_to<stdex::partial_ordering>(values::operation(
          c,
          values::fixed_value_of<decltype(internal::get_descriptor_hash_code(std::declval<RA>()))>{},
          values::fixed_value_of<decltype(internal::get_descriptor_hash_code(std::declval<RB>()))>{}));
        if constexpr (values::fixed_value_of_v<decltype(cmp)> == stdex::partial_ordering::less or
            values::fixed_value_of_v<decltype(cmp)> == stdex::partial_ordering::greater)
          return values::fixed_partial_ordering_unordered {};
        else
          return cmp;
      }
      else
      {
        return values::fixed_partial_ordering_unordered {};
      }
    }
    else if constexpr (not collections::sized<A>)
    {
      using RA = stdex::ranges::range_value_t<A>;
      if constexpr (values::fixed_value_compares_with<dimension_of<RA>, 0>)
      {
        return compare_three_way(Dimensions<0>{}, b, c);
      }
      else
      {
        auto offset = std::integral_constant<std::size_t, 0>{};
        auto extent = values::operation(std::plus{}, collections::get_size(b), std::integral_constant<std::size_t, 1>{});
        return compare_three_way(collections::views::slice(a, offset, extent), b, c);
      }
    }
    else if constexpr (not collections::sized<B>)
    {
      using RB = stdex::ranges::range_value_t<B>;
      if constexpr (values::fixed_value_compares_with<dimension_of<RB>, 0>)
      {
        return compare_three_way(a, Dimensions<0>{}, c);
      }
      else
      {
        auto offset = std::integral_constant<std::size_t, 0>{};
        auto extent = values::operation(std::plus{}, collections::get_size(a), std::integral_constant<std::size_t, 1>{});
        return compare_three_way(a, collections::views::slice(b, offset, extent), c);
      }
    }

    else if constexpr (collections::size_of_v<A> != stdex::dynamic_extent and collections::size_of_v<B> != stdex::dynamic_extent)
    {
      if constexpr (collections::size_of_v<A> == 0 or collections::size_of_v<B> == 0)
        return values::cast_to<stdex::partial_ordering>(values::operation(c, collections::size_of<A>{}, collections::size_of<B>{}));
      else
        return detail::compare_three_way_fixed(a, b, c);
    }
    else if constexpr (collections::size_of_v<A> != stdex::dynamic_extent) // collections::size_of_v<B> == stdex::dynamic_extent
    {
      bool size_b_is_zero = values::to_value_type(collections::get_size(b)) == 0;
      if constexpr (collections::size_of_v<A> == 0)
        return size_b_is_zero ? stdex::partial_ordering::equivalent : stdex::partial_ordering::less;
      else if (size_b_is_zero)
        return stdex::partial_ordering::greater;
      else
        return detail::compare_three_way_impl(collections::views::all(std::forward<A>(a)), b, c);
    }
    else if constexpr (collections::size_of_v<B> != stdex::dynamic_extent) // collections::size_of_v<A> == stdex::dynamic_extent
    {
      bool size_a_is_zero = values::to_value_type(collections::get_size(a)) == 0;
      if constexpr (collections::size_of_v<B> == 0)
        return size_a_is_zero ? stdex::partial_ordering::equivalent : stdex::partial_ordering::greater;
      else if (size_a_is_zero)
        return stdex::partial_ordering::less;
      else
        return detail::compare_three_way_impl(a, collections::views::all(std::forward<B>(b)), c);
    }
    else
    {
      bool size_a_is_zero = values::to_value_type(collections::get_size(a)) == 0;
      bool size_b_is_zero = values::to_value_type(collections::get_size(b)) == 0;
      if (size_a_is_zero)
        return size_b_is_zero ? stdex::partial_ordering::equivalent : stdex::partial_ordering::less;
      else if (size_b_is_zero)
        return stdex::partial_ordering::greater;
      else
        return detail::compare_three_way_impl(a, b, c);
    }
  }

}

#endif
