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
 * \brief Definition for \ref patterns::collection_patterns_compare_with_dimension.
 */

#ifndef OPENKALMAN_PATTERNS_COLLECTION_PATTERNS_COMPARE_WITH_DIMENSION_HPP
#define OPENKALMAN_PATTERNS_COLLECTION_PATTERNS_COMPARE_WITH_DIMENSION_HPP

#include "collections/collections.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/traits/pattern_collection_element.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct has_common_collection_type : std::false_type {};

    template<typename T>
    struct has_common_collection_type<T, std::void_t<typename collections::common_collection_type<T>::type>> : std::true_type {};
#endif


    template<typename T, std::size_t dim, auto comp, std::size_t N, applicability b, typename = std::make_index_sequence<N>>
    struct collection_patterns_compare_with_dimension_fixed_size : std::false_type {};

    template<typename T, std::size_t dim, auto comp, std::size_t N, applicability b, std::size_t...i>
    struct collection_patterns_compare_with_dimension_fixed_size<T, dim, comp, N, b, std::index_sequence<i...>>
      : std::bool_constant<(... and values::size_compares_with<
          dimension_of<pattern_collection_element_t<i, T>>,
          std::integral_constant<std::size_t, dim>,
          comp, b>)> {};


    template<typename T, std::size_t dim, auto comp, auto N, applicability b>
    constexpr bool
    collection_patterns_compare_with_dimension_impl()
    {
      if constexpr (values::fixed_value_compares_with<collections::size_of<T>, stdex::dynamic_extent, &stdex::is_neq>)
      {
#ifdef __cpp_concepts
        if constexpr (stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>)
#else
        if constexpr (N == std::size_t(values::unbounded_size))
#endif
          return collection_patterns_compare_with_dimension_fixed_size<T, dim, comp, collections::size_of_v<T>, b>::value;
        else
          return collection_patterns_compare_with_dimension_fixed_size<T, dim, comp, N, b>::value;
      }
#ifdef __cpp_concepts
      else if constexpr (requires { typename collections::common_collection_type<T>::type; })
#else
      else if constexpr (has_common_collection_type<T>::value)
#endif
      {
        return values::size_compares_with<
          dimension_of<collections::common_collection_type_t<T>>,
          std::integral_constant<std::size_t, dim>, comp, b>;
      }
      else
      {
        return b == applicability::permitted;
      }
    }

  }


  /**
   * \brief Specifies that each element of a \ref pattern_collection T has dimension dim for the first N indices.
   * \details If N is greater than the size of T, T will effectively be padded with Dimensions<1>.
   * \tparam N Either \ref values::unbounded_size or an integer greater than 0.
   */
#ifdef __cpp_concepts
  template<
    typename T,
    std::size_t dim,
    auto comp = &stdex::is_eq,
    auto N = values::unbounded_size,
    applicability b = applicability::guaranteed>
  concept collection_patterns_compare_with_dimension =
    (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N > 0) and
#else
  template<typename T, std::size_t dim, auto comp = &stdex::is_eq, std::size_t N = values::unbounded_size, applicability b = applicability::guaranteed>
  constexpr inline bool collection_patterns_compare_with_dimension =
    (N == values::unbounded_size or N > 0) and
#endif
    pattern_collection<T> and
    detail::collection_patterns_compare_with_dimension_impl<T, dim, comp, N, b>();


}

#endif
