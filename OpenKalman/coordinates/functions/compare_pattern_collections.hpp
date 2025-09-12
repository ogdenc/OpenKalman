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
#include "coordinates/concepts/pattern_collection.hpp"
#include "coordinates/concepts/pattern_collection_compares_with.hpp"
#include "coordinates/descriptors/Dimensions.hpp"
#include "coordinates/functions/compare.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<auto comp, std::size_t i = 0_uz, typename A, typename B>
    constexpr auto
    compare_pattern_collections_a_tail(const A& a, const B& b)
    {
      constexpr auto ix = std::integral_constant<std::size_t, i>{};
      if constexpr (i < collections::size_of_v<A>)
        return compare<comp>(collections::get(a, ix), Dimensions<1>{}) and
          compare_pattern_collections_a_tail<comp, i + 1>(a, b);
      else
        return std::true_type {};
    }


    template<auto comp, std::size_t i = 0_uz, typename A, typename B>
    constexpr auto
    compare_pattern_collections_b_tail(const A& a, const B& b)
    {
      constexpr auto ix = std::integral_constant<std::size_t, i>{};
      if constexpr (i < collections::size_of_v<B>)
        return compare<comp>(Dimensions<1>{}, collections::get(b, ix)) and
          compare_pattern_collections_b_tail<comp, i + 1>(a, b);
      else
        return std::true_type {};
    }


    template<auto comp, std::size_t i = 0_uz, typename A, typename B>
    constexpr auto
    compare_pattern_collections_iter_a(const A& a, const B& b)
    {
      constexpr auto ix = std::integral_constant<std::size_t, i>{};
      std::size_t size_b = collections::get_size(b);
      if constexpr (i < collections::size_of_v<A>)
      {
        if (i < size_b)
          return compare<comp>(collections::get(a, ix), collections::get(b, ix)) and
            compare_pattern_collections_iter_a<comp, i + 1>(a, b);
        else
          return compare_pattern_collections_a_tail<comp, i>(a, b);
      }
      else
      {
        for (std::size_t j = i; j < size_b; ++j)
          if (not compare<comp>(Dimensions<1>{}, collections::get(b, j))) return false;
        return true;
      }
    }


    template<auto comp, std::size_t i = 0_uz, typename A, typename B>
    constexpr auto
    compare_pattern_collections_iter_b(const A& a, const B& b)
    {
      constexpr auto ix = std::integral_constant<std::size_t, i>{};
      std::size_t size_a = collections::get_size(a);
      if constexpr (i < collections::size_of_v<B>)
      {
        if (i < size_a)
          return compare<comp>(collections::get(a, ix), collections::get(b, ix)) and
            compare_pattern_collections_iter_b<comp, i + 1>(a, b);
        else
          return compare_pattern_collections_b_tail<comp, i>(a, b);
      }
      else
      {
        for (std::size_t j = i; j < size_a; ++j)
          if (not compare<comp>(collections::get(a, j), Dimensions<1>{})) return false;
        return true;
      }
    }


    template<auto comp, std::size_t i = 0_uz, typename A, typename B>
    constexpr auto
    compare_pattern_collections_iter(const A& a, const B& b)
    {
      constexpr auto ix = std::integral_constant<std::size_t, i>{};
      if constexpr (i < collections::size_of_v<A>)
      {
        if constexpr (i < collections::size_of_v<B>)
          return compare<comp>(collections::get(a, ix), collections::get(b, ix)) and
            compare_pattern_collections_iter<comp, i + 1>(a, b);
        else
          return compare_pattern_collections_a_tail<comp, i>(a, b);
      }
      else
      {
        return compare_pattern_collections_b_tail<comp, i>(a, b);
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
  template<auto comp = &stdcompat::is_eq, pattern_collection A, pattern_collection B> requires
    std::is_invocable_r_v<bool, decltype(comp), stdcompat::partial_ordering>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<auto comp = &stdcompat::is_eq, typename A, typename B, std::enable_if_t<
    pattern_collection<A> and pattern_collection<B> and
    std::is_invocable_r_v<bool, decltype(comp), stdcompat::partial_ordering>, int> = 0>
constexpr auto
#endif
  compare_pattern_collections(const A& a, const B& b)
  {
    if constexpr (pattern_collection_compares_with<A, B, comp, applicability::guaranteed>)
    {
      return std::true_type {};
    }
    else if constexpr (not pattern_collection_compares_with<A, B, comp, applicability::permitted> or
      not collections::sized<A> or not collections::sized<B>)
    {
      return std::false_type {};
    }
    else if constexpr (collections::size_of_v<A> == dynamic_size and collections::size_of_v<B> == dynamic_size)
    {
      std::size_t size_a = collections::get_size(a);
      std::size_t size_b = collections::get_size(b);
      std::size_t i = 0;
      for (; i < size_a and i < size_b; ++i)
        if (not compare<comp>(collections::get(a, i), collections::get(b, i))) return false;
      for (; i < size_a; ++i)
        if (not compare<comp>(collections::get(a, i), Dimensions<1>{})) return false;
      for (; i < size_b; ++i)
        if (not compare<comp>(Dimensions<1>{}, collections::get(b, i))) return false;
      return true;
    }
    else if constexpr (collections::size_of_v<B> == dynamic_size)
    {
      return detail::compare_pattern_collections_iter_a<comp>(a, b);
    }
    else if constexpr (collections::size_of_v<A> == dynamic_size)
    {
      return detail::compare_pattern_collections_iter_b<comp>(a, b);
    }
    else
    {
      return detail::compare_pattern_collections_iter<comp>(a, b);
    }
  }

}

#endif
