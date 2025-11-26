/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref pattern_collection_compares_with.
 */

#ifndef OPENKALMAN_COORDINATE_PATTERN_COLLECTION_COMPARES_WITH_HPP
#define OPENKALMAN_COORDINATE_PATTERN_COLLECTION_COMPARES_WITH_HPP

#include "collections/collections.hpp"
#include "pattern_collection.hpp"
#include "compares_with.hpp"
#include "coordinates/descriptors/Dimensions.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<typename T, typename PU, auto comp, applicability a, std::size_t i = 0>
    constexpr bool
    pattern_collection_compares_with_iter_T()
    {
      if constexpr (i < collections::size_of_v<T>)
      {
        using Ti = std::decay_t<collections::collection_element_t<i, T>>;
        if constexpr (compares_with<Ti, PU, comp, a>)
          return pattern_collection_compares_with_iter_T<T, PU, comp, a, i + 1>();
        else
          return false;
      }
      else
      {
        return true;
      }
    }


    template<typename PT, typename U, auto comp, applicability a, std::size_t i = 0>
    constexpr bool
    pattern_collection_compares_with_iter_U()
    {
      if constexpr (i < collections::size_of_v<U>)
      {
        using Ui = std::decay_t<collections::collection_element_t<i, U>>;
        if constexpr (compares_with<PT, Ui, comp, a>)
          return pattern_collection_compares_with_iter_U<PT, U, comp, a, i + 1>();
        else
          return false;
      }
      else
      {
        return true;
      }
    }


    template<typename T, typename U, auto comp, applicability a, std::size_t i = 0>
    constexpr bool
    pattern_collection_compares_with_iter()
    {
      if constexpr (i < collections::size_of_v<T>)
      {
        if constexpr (i < collections::size_of_v<U>)
        {
          using Ti = std::decay_t<collections::collection_element_t<i, T>>;
          using Ui = std::decay_t<collections::collection_element_t<i, U>>;
          if constexpr (compares_with<Ti, Ui, comp, a>)
            return pattern_collection_compares_with_iter<T, U, comp, a, i + 1>();
          else
            return false;
        }
        else
        {
          return pattern_collection_compares_with_iter_T<T, Dimensions<1>, comp, a, i>();
        }
      }
      else if constexpr (i < collections::size_of_v<U>)
      {
        return pattern_collection_compares_with_iter_U<Dimensions<1>, U, comp, a, i>();
      }
      else
      {
        return true;
      }
    }


    template<typename T, typename U, auto comp, applicability a>
    constexpr bool
    pattern_collection_compares_with_impl()
    {
      constexpr bool fixed_t = collections::sized<T> and not values::fixed_value_compares_with<collections::size_of<T>, stdex::dynamic_extent>;
      constexpr bool fixed_u = collections::sized<U> and not values::fixed_value_compares_with<collections::size_of<U>, stdex::dynamic_extent>;
      if constexpr (fixed_t and fixed_u)
      {
        return detail::pattern_collection_compares_with_iter<T, U, comp, a>();
      }
      else if constexpr (fixed_t)
      {
        return detail::pattern_collection_compares_with_iter_T<T, stdex::ranges::range_value_t<U>, comp, a>();
      }
      else if constexpr (fixed_u)
      {
        return detail::pattern_collection_compares_with_iter_U<stdex::ranges::range_value_t<T>, U, comp, a>();
      }
      else
      {
        return compares_with<stdex::ranges::range_value_t<T>, stdex::ranges::range_value_t<U>, comp, a>;
      }
    }

  }


  /**
   * \brief Compares a two \ref coordinates::pattern_collection objects.
   * \details Every \ref coordinate_list in the set must be potentially equivalent to every other \ref coordinate_list
   * in the set. Trailing 1D patterns are ignored.
   * \tparam comp A consteval-callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, typename U, auto comp = &stdex::is_eq, applicability a = applicability::guaranteed>
#ifdef __cpp_concepts
  concept pattern_collection_compares_with =
#else
  constexpr bool pattern_collection_compares_with =
#endif
    pattern_collection<T> and pattern_collection<U> and
    std::is_invocable_r_v<bool, decltype(comp), stdex::partial_ordering> and
    detail::pattern_collection_compares_with_impl<T, U, comp, a>();


}

#endif
