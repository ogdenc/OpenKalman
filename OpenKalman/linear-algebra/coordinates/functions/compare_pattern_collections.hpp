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
 * \brief Definition of \ref coordinates::compare_pattern_collections.
 */

#ifndef OPENKALMAN_COORDINATES_COMPARE_PATTERN_COLLECTIONS_HPP
#define OPENKALMAN_COORDINATES_COMPARE_PATTERN_COLLECTIONS_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<auto comp, std::size_t sizea, std::size_t sizeb, std::size_t i = 0_uz, typename A, typename B>
    constexpr auto
    compare_pattern_collections_fixed(const A& a, const B& b)
    {
      if constexpr (i < sizea or i < sizeb)
      {
        auto c = [](const A& a, const B& b)
        {
          auto ix = std::integral_constant<std::size_t, i>{};
          if constexpr (i < sizea and i < sizeb)
          {
            auto ai = collections::get(a, ix);
            auto bi = collections::get(b, ix);
            if constexpr (compares_with<decltype(ai), decltype(bi), comp>) return std::true_type{};
            else return std::invoke(comp, compare(ai, bi));
          }
          else if constexpr (i < sizea)
          {
            auto ai = collections::get(a, ix);
            if constexpr (compares_with<decltype(ai), Dimensions<1>, comp>) return std::true_type{};
            else return std::invoke(comp, compare(ai, Dimensions<1>{}));
          }
          else // if constexpr (i < sizeb)
          {
            auto bi = collections::get(b, ix);
            if constexpr (compares_with<Dimensions<1>, decltype(bi), comp>) return std::true_type{};
            else return std::invoke(comp, compare(Dimensions<1>{}, bi));
          }
        }(a, b);

        if constexpr (values::fixed_number_compares_with<decltype(c), true>)
          return compare_pattern_collections_fixed<comp, sizea, sizeb, i + 1>(a, b);
        else if constexpr (values::fixed_number_compares_with<decltype(c), false>)
          return std::false_type{};
        else
        {
          if (c) return static_cast<bool>(compare_pattern_collections_fixed<comp, sizea, sizeb, i + 1>(a, b));
          else return false;
        }
      }
      else
      {
        return std::true_type{};
      }
    }
  }


  /**
   * \brief Compare each element of two \ref pattern_collection objects lexicographically.
   * \detail Performs a \ref compare operation on each element of the collection.
   * If the patterns are different sizes, trailing Dimensions<1> patterns are added to the smaller one before the comparison.
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
#ifdef __cpp_concepts
  template<auto comp = &stdcompat::is_eq, pattern_collection A, pattern_collection B>
    requires collections::sized<A> and collections::sized<B>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<auto comp = &stdcompat::is_eq, typename A, typename B, std::enable_if_t<
    pattern_collection<A> and pattern_collection<B> and collections::sized<A> and collections::sized<B>, int> = 0>
  constexpr auto
#endif
  compare_pattern_collections(const A& a, const B& b)
  {
    if constexpr (collections::size_of_v<A> == dynamic_size or collections::size_of_v<B> == dynamic_size)
    {
      std::size_t size_a = collections::get_size(a);
      std::size_t size_b = collections::get_size(b);
      std::size_t i = 0;
      for (; i < std::min(size_a, size_b); ++i)
        if (not std::invoke(comp, compare(collections::get(collections::views::all(a), i), collections::get(collections::views::all(b), i)))) return false;
      for (; i < size_a; ++i)
        if (not std::invoke(comp, compare(collections::get(collections::views::all(a), i), Dimensions<1>{}))) return false;
      for (; i < size_b; ++i)
        if (not std::invoke(comp, compare(Dimensions<1>{}, collections::get(collections::views::all(b), i)))) return false;
      return true;
    }
    else
    {
      return detail::compare_pattern_collections_fixed<comp, collections::size_of_v<A>, collections::size_of_v<B>>(a, b);
    }
  }

}

#endif
