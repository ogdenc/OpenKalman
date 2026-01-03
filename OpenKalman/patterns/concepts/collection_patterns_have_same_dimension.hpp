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
 * \brief Definition for \ref patterns::collection_patterns_have_same_dimension.
 */

#ifndef OPENKALMAN_PATTERNS_COLLECTION_PATTERNS_HAVE_SAME_DIMENSION_HPP
#define OPENKALMAN_PATTERNS_COLLECTION_PATTERNS_HAVE_SAME_DIMENSION_HPP

#include "collections/collections.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/concepts/fixed_pattern.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/traits/pattern_collection_element.hpp"
#include "patterns/functions/get_common_pattern_collection_dimension.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct patt_dim_is_fixed_2 : std::false_type {};

    template<typename T>
    struct patt_dim_is_fixed_2<T, std::enable_if_t<fixed_pattern<typename T::type>>>
      : std::true_type {};
#endif


    template<typename T, std::size_t N, std::size_t i = 0>
    constexpr std::size_t
    best_comparison_dim()
    {
      if constexpr (i < N)
      {
        using P = pattern_collection_element_t<i, T>;
        if constexpr (fixed_pattern<P>) return dimension_of_v<P>;
        else return best_comparison_dim<T, N, i + 1>();
      }
      else
      {
        return stdex::dynamic_extent;
      }
    }


    template<typename T, std::size_t N, applicability b, typename = std::make_index_sequence<N>>
    struct collection_dim_comp_fixed : std::false_type {};

    template<typename T, std::size_t N, applicability b, std::size_t...i>
    struct collection_dim_comp_fixed<T, N, b, std::index_sequence<i...>>
      : std::bool_constant<(... and values::size_compares_with<
          dimension_of<pattern_collection_element_t<i, T>>,
          std::integral_constant<std::size_t, best_comparison_dim<T, N>()>,
          &stdex::is_eq, b>)> {};


    template<typename T, auto N, applicability b>
    constexpr bool
    collection_patterns_have_same_dimension_impl()
    {
      constexpr bool n_lt_2 = []{
#ifdef __cpp_concepts
          if constexpr (stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) return false;
#else
          if constexpr (N == std::size_t(values::unbounded_size)) return false;
#endif
          else return N < 2;
        }();

      if constexpr (values::fixed_value_compares_with<collections::size_of<T>, stdex::dynamic_extent, &stdex::is_neq>)
      {
#ifdef __cpp_concepts
        if constexpr (stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>)
#else
        if constexpr (N == std::size_t(values::unbounded_size))
#endif
          return collection_dim_comp_fixed<T, collections::size_of_v<T>, b>::value;
        else
          return n_lt_2 or collection_dim_comp_fixed<T, N, b>::value;
      }
      else
      {
        return n_lt_2 or
#ifdef __cpp_concepts
          requires { requires fixed_pattern<collections::common_collection_type_t<T>>; } or
#else
          patt_dim_is_fixed_2<collections::common_collection_type<T>>::value or
#endif
          b == applicability::permitted;
      }
    }

  }


  /**
   * \brief Specifies that the first N elements of a \ref pattern_collection have the same dimensions.
   * \details If N exceeds the size of T, T will be padded with Dimensions<1>.
   * If N == values::unbounded_size, all elements will be compared.
   * Note that the result will always be true if the number of compared elements is 1.
   * \tparam N Either an integer greater than 0 or \ref values::unbounded_size
   */
#ifdef __cpp_concepts
  template<typename T, auto N = values::unbounded_size, applicability b = applicability::guaranteed>
  concept collection_patterns_have_same_dimension =
    (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N > 0) and
#else
  template<typename T, std::size_t N = values::unbounded_size, applicability b = applicability::guaranteed>
  constexpr inline bool collection_patterns_have_same_dimension =
    (N == values::unbounded_size or N > 0) and
#endif
    pattern_collection<T> and
    detail::collection_patterns_have_same_dimension_impl<T, N, b>();


}

#endif
