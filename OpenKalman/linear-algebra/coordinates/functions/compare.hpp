/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref coordinates::compare.
 */

#ifndef OPENKALMAN_COORDINATES_COMPARE_HPP
#define OPENKALMAN_COORDINATES_COMPARE_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_collection_element.hpp"
#include "linear-algebra/coordinates/functions/comparison-operators.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<std::size_t ai = 0, std::size_t bi = 0, std::size_t abank = 0, std::size_t bbank = 0, typename A, typename B, typename C>
    constexpr stdcompat::partial_ordering
    compare_fixed(const A& a, const B& b, C c)
    {
      if constexpr (ai < collections::size_of_v<A> and bi < collections::size_of_v<B>)
      {
        using Ai = std::tuple_element_t<ai, A>;
        using Bi = std::tuple_element_t<bi, B>;
        constexpr bool ae = euclidean_pattern<Ai>, be = euclidean_pattern<Bi>;
        if constexpr (ae and be) return compare_fixed<ai + 1, bi + 1, abank + dimension_of_v<Ai>, bbank + dimension_of_v<Bi>>(a, b, c);
        else if constexpr (ae) return compare_fixed<ai + 1, bi, abank + dimension_of_v<Ai>, bbank>(a, b, c);
        else if constexpr (be) return compare_fixed<ai, bi + 1, abank, bbank + dimension_of_v<Bi>>(a, b, c);
        else
        {
          if constexpr (abank == bbank)
            if (internal::get_descriptor_hash_code(OpenKalman::internal::generalized_std_get<ai>(a)) ==
                internal::get_descriptor_hash_code(OpenKalman::internal::generalized_std_get<bi>(b)))
              return compare_fixed<ai + 1, bi + 1>(a, b, c);
          return stdcompat::partial_ordering::unordered;
        }
      }
      else if constexpr (ai < collections::size_of_v<A>) // bi >= collections::size_of_v<B>
      {
        using Ai = std::tuple_element_t<ai, A>;
        if (euclidean_pattern<Ai>) return compare_fixed<ai + 1, bi, abank + dimension_of_v<Ai>, bbank>(a, b, c);
        if (values::to_number(abank) >= values::to_number(bbank)) return stdcompat::partial_ordering::greater;
        return stdcompat::partial_ordering::unordered;
      }
      else if constexpr (bi < collections::size_of_v<B>) // ai >= collections::size_of_v<A>
      {
        using Bi = std::tuple_element_t<bi, B>;
        if (euclidean_pattern<Bi>) return compare_fixed<ai, bi + 1, abank, bbank + dimension_of_v<Bi>>(a, b, c);
        if (values::to_number(abank) <= values::to_number(bbank)) return stdcompat::partial_ordering::less;
        return stdcompat::partial_ordering::unordered;
      }
      else
      {
        // not ai_in_range and not bi_in_range:
        return stdcompat::invoke(c, values::to_number(abank), values::to_number(bbank));
      }
    }


    template<typename A, typename B, typename C>
    constexpr stdcompat::partial_ordering
    compare_impl(const A& a, const B& b, C c, std::size_t ai = 0, std::size_t bi = 0, std::size_t abank = 0, std::size_t bbank = 0)
    {
      bool ai_in_range = ai < values::to_number(get_size(a));
      bool bi_in_range = bi < values::to_number(get_size(b));

      if (ai_in_range and bi_in_range)
      {
        auto a_i = internal::get_descriptor_collection_element(a, ai);
        auto b_i = internal::get_descriptor_collection_element(b, bi);
        bool ae = get_is_euclidean(a_i), be = get_is_euclidean(b_i);
        if (ae or be) return compare_impl(a, b, c, (ae ? ai + 1 : ai), (be ? bi + 1 : bi),
          (ae ? abank + get_dimension(a_i) : abank), (be ? bbank + get_dimension(b_i) : bbank));
        if (internal::get_descriptor_hash_code(a_i) == internal::get_descriptor_hash_code(b_i) and abank == bbank) return compare_impl(a, b, c, ai + 1, bi + 1);
        return stdcompat::partial_ordering::unordered;
      }
      if (ai_in_range) // not bi_in_range
      {
        auto a_i = internal::get_descriptor_collection_element(a, ai);
        if (get_is_euclidean(a_i)) return compare_impl(a, b, c, ai + 1, bi, abank + get_dimension(a_i), bbank);
        if (abank >= bbank) return stdcompat::partial_ordering::greater;
        return stdcompat::partial_ordering::unordered;
      }
      if (bi_in_range) // not ai_in_range
      {
        auto b_i = internal::get_descriptor_collection_element(b, bi);
        if (get_is_euclidean(b_i)) return compare_impl(a, b, c, ai, bi + 1, abank, bbank + get_dimension(b_i));
        if (abank <= bbank) return stdcompat::partial_ordering::less;
        return stdcompat::partial_ordering::unordered;
      }
      // not ai_in_range and not bi_in_range:
      return stdcompat::invoke(c, abank, bbank);
    }

  }


  /**
   * \brief Compare two \ref coordinates::pattern objects lexicographically.
   * \detail Consecutive \ref euclidean_pattern arguments are consolidated before the comparison occurs.
   * \tparam Comparison A callable comparison function compatible with std::partial_ordering, such as std::compare_three_way
   */
#ifdef __cpp_concepts
  template<pattern A, pattern B, typename Comparison = stdcompat::compare_three_way>
    requires (descriptor<A> or collections::sized<A>) and (descriptor<B> or collections::sized<B>)
#else
  template<typename A, typename B, typename Comparison = stdcompat::compare_three_way, std::enable_if_t<
    pattern<A> and pattern<B> and (descriptor<A> or collections::sized<A>) and (descriptor<B> or collections::sized<B>), int> = 0>
#endif
  constexpr stdcompat::partial_ordering
  compare(const A& a, const B& b, Comparison c = {})
  {
    if constexpr (descriptor<A> and descriptor<B> and stdcompat::invocable<Comparison, A, B>)
    {
      return stdcompat::invoke(c, a, b);
    }
    else if constexpr (descriptor<A>)
    {
      return compare(std::tuple {a}, b, c);
    }
    else if constexpr (descriptor<B>)
    {
      return compare(a, std::tuple {b}, c);
    }
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>)
    {
      if constexpr (euclidean_pattern<A> and euclidean_pattern<B>)
        return stdcompat::invoke(c, dimension_of_v<A>, dimension_of_v<B>);
      else if constexpr (collections::size_of_v<A> == 0) return stdcompat::partial_ordering::less;
      else if constexpr (collections::size_of_v<B> == 0) return stdcompat::partial_ordering::greater;
      else return detail::compare_fixed(a, b, c);
    }
    else
    {
      if (get_is_euclidean(a) and get_is_euclidean(b))
        return stdcompat::invoke(c, values::to_number(get_dimension(a)), values::to_number(get_dimension(b)));
      if (values::to_number(get_dimension(a)) == 0) return stdcompat::partial_ordering::less;
      if (values::to_number(get_dimension(b)) == 0) return stdcompat::partial_ordering::greater;
      return detail::compare_impl(collections::views::all(a), collections::views::all(b), c);
    }
  }

} // namespace OpenKalman::coordinates

#endif
