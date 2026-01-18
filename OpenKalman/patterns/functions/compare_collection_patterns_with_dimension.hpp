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
 * \brief Definition for \ref patterns::compare_collection_patterns_with_dimension.
 */

#ifndef OPENKALMAN_PATTERNS_COMPARE_COLLECTION_PATTERNS_WITH_DIMENSION_HPP
#define OPENKALMAN_PATTERNS_COMPARE_COLLECTION_PATTERNS_WITH_DIMENSION_HPP

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
    struct has_fixed_patt_dim : std::false_type {};

    template<typename T>
    struct has_fixed_patt_dim<T, std::enable_if_t<fixed_pattern<typename T::type>>>
      : std::true_type {};
#endif


    template<auto comp>
    constexpr auto
    do_compare = [](const auto& a, const auto& b) {
      return stdex::invoke(comp, stdex::compare_three_way{}(a, b));
    };


    template<std::size_t n, auto comp, std::size_t i = 0, typename T, typename D>
    constexpr auto
    compare_collection_patterns_with_dimension_impl(const T& t, const D& d)
    {
      if constexpr (i < n)
        return values::operation(
          std::logical_and{},
          values::operation(do_compare<comp>, get_dimension(get_pattern<i>(t)), d),
          compare_collection_patterns_with_dimension_impl<n, comp, i + 1>(t, d));
      else
        return std::true_type {};
    }

  }


  /**
   * \brief Compares the dimensions of the first N elements of a \ref pattern_collection with a particular value.
   * \details If N exceeds the size of T, T will effectively be padded with Dimensions<1>.
   * If N == values::unbounded_size, all elements will be compared.
   * \tparam N Either \ref values::unbounded_size or an integer greater than 0.
   */
#ifdef __cpp_concepts
  template<auto comp = &stdex::is_eq, auto N = values::unbounded_size, pattern_collection T, values::index D> requires
    (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N > 0)
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<auto comp = &stdex::is_eq, std::size_t N = values::unbounded_size, typename T, typename D, std::enable_if_t<
    (N == values::unbounded_size or N > 0) and pattern_collection<T> and values::index<D>, int> = 0>
  constexpr auto
#endif
  compare_collection_patterns_with_dimension(const T& t, const D& d)
  {
    constexpr bool has_fixed_common =
#ifdef __cpp_concepts
    requires { requires fixed_pattern<collections::common_collection_type_t<T>>; };
#else
    detail::has_fixed_patt_dim<collections::common_collection_type<T>>::value;
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
          return values::operation(detail::do_compare<comp>, dimension_of<collections::common_collection_type_t<T>>{}, d);
        else
          return detail::compare_collection_patterns_with_dimension_impl<values::fixed_value_of_v<n_type>, comp>(t, d);
      }
      else
      {
        for (std::size_t i = 0; i < n; ++i)
          if (not detail::do_compare<comp>(get_dimension(get_pattern(t, i)), values::to_value_type(d))) return false;
        return true;
      }
    }
    else if constexpr (has_fixed_common)
    {
      return values::operation(detail::do_compare<comp>, dimension_of<collections::common_collection_type_t<T>>{}, d);
    }
    else
    {
      return std::false_type {};
    }

  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
template<std::size_t dim, auto comp = &stdex::is_eq, auto N = values::unbounded_size, pattern_collection T> requires
  (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
  (not values::integral<decltype(N)> or N > 0)
constexpr OpenKalman::internal::boolean_testable auto
#else
template<std::size_t dim, auto comp = &stdex::is_eq, std::size_t N = values::unbounded_size, typename T, std::enable_if_t<
  (N == values::unbounded_size or N > 0) and pattern_collection<T>, int> = 0>
constexpr auto
#endif
compare_collection_patterns_with_dimension(const T& t)
  {
    return compare_collection_patterns_with_dimension<comp, N>(t, std::integral_constant<std::size_t, dim>{});
  }


}

#endif
