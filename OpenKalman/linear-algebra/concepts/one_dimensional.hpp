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
 * \brief Definition for \ref one_dimensional.
 */

#ifndef OPENKALMAN_ONE_DIMENSIONAL_HPP
#define OPENKALMAN_ONE_DIMENSIONAL_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/dimension_size_of_index_is.hpp"
#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/traits/pattern_collection_type_of.hpp"

namespace OpenKalman
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, auto N, applicabilty b, typename = void>
    struct one_dimensional_impl : std::false_type {};

    template<typename T, auto N, applicabilty b>
    struct one_dimensional_impl<T, N, b, std::enable_if_t<
      patterns::collection_patterns_compare_with_dimension<typename pattern_collection_type_of<T>::type, 1, &stdex::is_eq, N, b>
      > : std::true_type {};
#endif


    template<typename T, std::size_t N, typename = std::make_index_sequence<N>>
    struct any_1d_index_impl : std::false_type {};

    template<typename T, std::size_t N, std::size_t...i>
    struct any_1d_index_impl<T, N, std::index_sequence<i...>>
      : std::bool_constant<(... or dimension_size_of_index_is<T, i, 1>)> {};


    template<typename T, auto N>
    constexpr bool
    any_1d_index()
    {
#ifdef __cpp_concepts
      if constexpr (stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>)
#else
      if constexpr (N == std::size_t(values::unbounded_size))
#endif
        return any_1d_index_impl<T, collections::size_of_v<T>>::value;
      else
        return any_1d_index_impl<T, N>::value;
    }
  }


  /**
   * \brief Specifies that a type is one-dimensional in every index.
   * \details Each index need not have an equivalent \ref patterns::pattern "pattern".
   */
#ifdef __cpp_concepts
  template<typename T, auto N = values::unbounded_size, applicability b = applicability::guaranteed>
  concept one_dimensional =
    indexible<T> and
    (values::size<decltype(N)> or (values::integral<decltype(N)> and N >= 0)) and
    (patterns::collection_patterns_compare_with_dimension<pattern_collection_type_of_t<T>, 1, &stdex::is_eq, N, b> or
      (square_shaped<T, N, b> and detail::any_1d_index<T, N>()));
#else
  template<typename T, std::size_t N = values::unbounded_size, applicability b = applicability::guaranteed>
  constexpr inline bool one_dimensional =
    detail::one_dimensional_impl<T, N, b>::value or
    (square_shaped<T, N, b> and detail::any_1d_index<T, N>());
#endif


}

#endif
