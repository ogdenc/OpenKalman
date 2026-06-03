/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
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
    template<typename T, std::size_t N, typename = std::make_index_sequence<N>>
    struct any_1d_index : std::false_type {};

    template<typename T, std::size_t N, std::size_t...i>
    struct any_1d_index<T, N, std::index_sequence<i...>>
      : std::bool_constant<(... or dimension_size_of_index_is<T, i, 1>)> {};


#ifndef __cpp_concepts
    template<typename T, typename Size>
    constexpr bool
    one_dimensional_impl()
    {
      if constexpr (values::fixed<Size>)
      {
        constexpr std::size_t s = values::fixed_value_of_v<Size>;
        if constexpr (s == 1) return square_shaped<T, 1>;
        else return square_shaped<T, s> and any_1d_index<T, s>::value;
      }
      else return false;
    }

    template<typename T, applicability b, typename = void>
    struct pattern_compare_impl : std::false_type {};

    template<typename T, applicability b>
    struct pattern_compare_impl<T, b, std::void_t<typename pattern_collection_type_of<T>::type>>
      : std::bool_constant<(
        patterns::collection_patterns_compare_with_dimension<pattern_collection_type_of_t<T>, 1, &stdex::is_eq, values::unbounded_size, b> or
        (b == applicability::guaranteed and detail::one_dimensional_impl<T, collections::size_of<pattern_collection_type_of_t<T>>>()))> {};
#endif
  }


  /**
   * \brief Specifies that a type is one-dimensional in all of its indices.
   * \details Note that the dimension of any indices greater than \ref index_count_v<T> will naturally be 1.
   */
#ifdef __cpp_concepts
  template<typename T, applicability b = applicability::guaranteed>
  concept one_dimensional =
    indexible<T> and
    (patterns::collection_patterns_compare_with_dimension<pattern_collection_type_of_t<T>, 1, &stdex::is_eq, values::unbounded_size, b> or
      (b == applicability::guaranteed and
        square_shaped<T, collections::size_of_v<pattern_collection_type_of_t<T>>> and
        (collections::size_of_v<pattern_collection_type_of_t<T>> == 1 or
          detail::any_1d_index<T, collections::size_of_v<pattern_collection_type_of_t<T>>>::value)));
#else
  template<typename T, applicability b = applicability::guaranteed>
  constexpr inline bool one_dimensional =
    indexible<T> and detail::pattern_compare_impl<T, b>::value;
#endif


}

#endif
