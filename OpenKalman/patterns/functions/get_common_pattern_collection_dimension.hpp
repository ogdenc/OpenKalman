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
 * \brief Definition for \ref patterns::get_common_pattern_collection_dimension.
 */

#ifndef OPENKALMAN_PATTERNS_GET_COMMON_PATTERN_COLLECTION_DIMENSION_HPP
#define OPENKALMAN_PATTERNS_GET_COMMON_PATTERN_COLLECTION_DIMENSION_HPP

#include <optional>
#include "collections/collections.hpp"
#include "patterns/functions/get_dimension.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/concepts/fixed_pattern.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/functions/get_pattern.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct patt_dim_is_fixed : std::false_type {};

    template<typename T>
    struct patt_dim_is_fixed<T, std::enable_if_t<fixed_pattern<typename T::type>>>
      : std::true_type {};
#endif


    template<std::size_t n, std::size_t i = 0, typename T>
    constexpr auto
    collection_patterns_have_same_dimension_impl(const T& t)
    {
      auto d0 = get_dimension(get_pattern<i>(t));
      if constexpr (i + 1 < n)
      {
        auto tail = collection_patterns_have_same_dimension_impl<n, i + 1>(t);
        using tail_type = typename decltype(tail)::value_type;

        if constexpr (values::size_compares_with<decltype(d0), tail_type>)
          return std::optional {d0};
        else if (tail and *tail == values::to_value_type(d0))
          return std::optional<std::size_t> {d0};
        else
          return std::optional<std::size_t> {};
      }
      else return std::optional {d0};
    }

  }


  /**
   * \brief Queries whether the first N elements of a \ref pattern_collection have the same dimensions.
   * \details If N exceeds the size of T, T will be padded with Dimensions<1>.
   * If N == values::unbounded_size, all elements will be compared.
   * Note that the result will always be true if the number of compared elements is 1.
   * \tparam N Either \ref values::unbounded_size or an integer greater than 0.
   * \return a \ref std::optional containing the common dimension, if it exists.
   */
#ifdef __cpp_concepts
  template<auto N = values::unbounded_size, pattern_collection T> requires
    (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N > 0)
#else
  template<std::size_t N = values::unbounded_size, typename T, std::enable_if_t<
    (N == values::unbounded_size or N > 0) and pattern_collection<T>, int> = 0>
#endif
  constexpr auto
  get_common_pattern_collection_dimension(const T& t)
  {
    constexpr bool has_fixed_common =
#ifdef __cpp_concepts
    requires { requires fixed_pattern<collections::common_collection_type_t<T>>; };
#else
    detail::patt_dim_is_fixed<collections::common_collection_type<T>>::value;
#endif

    if constexpr (collections::sized<T>)
    {
      // n is the true number of elements that will be compared.
      auto n = [](const T& t){
#ifdef __cpp_concepts
        if constexpr (stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>)
#else
        if constexpr (N == std::size_t(values::unbounded_size))
#endif
          return collections::get_size(t);
        else
          return std::integral_constant<std::size_t, N>{};
      }(t);
      using n_type = decltype(n);

      if constexpr (values::fixed<n_type>)
      {
        if constexpr (has_fixed_common and values::size_compares_with<n_type, collections::size_of<T>, &stdex::is_lteq>)
          return std::optional {dimension_of<collections::common_collection_type_t<T>>{}};
        else
          return detail::collection_patterns_have_same_dimension_impl<values::fixed_value_of_v<n_type>>(t);
      }
      else
      {
        using Op = std::optional<std::size_t>;
        std::size_t d = get_dimension(get_pattern<0>(t));
        for (std::size_t i = 1; i < n; ++i) if (get_dimension(get_pattern(t, i)) != d) return Op{};
        return Op{d};
      }
    }
    else
    {
      if constexpr (has_fixed_common)
        return std::optional {dimension_of<collections::common_collection_type_t<T>>{}};
      else
        return std::optional<std::size_t> {};
    }

  }


}

#endif
