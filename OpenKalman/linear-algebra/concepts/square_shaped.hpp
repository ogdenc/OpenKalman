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
 * \brief Definition for \ref square_shaped.
 */

#ifndef OPENKALMAN_SQUARE_SHAPED_HPP
#define OPENKALMAN_SQUARE_SHAPED_HPP

#include "coordinates/coordinates.hpp"
#include "../interfaces/object_traits.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"

namespace OpenKalman
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, applicability b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, applicability b>
    struct is_explicitly_square<T, b, std::enable_if_t<interface::object_traits<stdex::remove_cvref_t<T>>::template is_square<b>>>
      : std::true_type {};
#endif

    template<typename T, std::size_t i>
    constexpr std::size_t
    best_square_index()
    {
      if constexpr (i == 0) return i;
      else if constexpr (coordinates::fixed_pattern<decltype(get_index_pattern<i - 1>(std::declval<T>()))>) return i - 1;
      else return best_square_index<T, i - 1>();
    }


    template<typename T, applicability b, std::size_t...is>
    constexpr auto
    square_shaped_fixed_index_count(std::index_sequence<is...>)
    {
      constexpr std::size_t best = best_square_index<T, sizeof...(is)>();
      using best_patt = decltype(get_index_pattern<best>(std::declval<T>()));
      return (... and (is == best or
        coordinates::compares_with<decltype(get_index_pattern<is>(std::declval<T>())), best_patt, stdex::is_eq, b>));
    }


    template<typename T, applicability b>
    constexpr auto
    square_shaped_impl()
    {
      if constexpr (not indexible<T>) // Only needed for c++17 mode
        return false;
      else if constexpr (index_count_v<T> == 1)
        return index_dimension_of_v<T, 0> == 1;
      else
        return detail::square_shaped_fixed_index_count<T, b>(std::make_index_sequence<index_count_v<T>>{});
    }

  }


  /**
   * \brief Specifies that an object is square (i.e., has equivalent \ref coordinates::pattern along each dimension).
   * \details Any trailing 1D Euclidean descriptors are disregarded. A vector must be one-dimensional.
   * A 0-by-0 matrix or tensor is considered to be square.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == applicability::guaranteed</code>: T is known at compile time to be square;
   * - if <code>b == applicability::permitted</code>: It is known at compile time that T <em>may</em> be square.
   */
  template<typename T, applicability b = applicability::guaranteed>
#ifdef __cpp_concepts
  concept square_shaped =
    indexible<T> and
    (not interface::is_square_defined_for<T, b> or
      interface::object_traits<stdex::remove_cvref_t<T>>::template is_square<b>) and
    (interface::is_square_defined_for<T, b> or
      detail::square_shaped_impl<T, b>());
#else
  constexpr bool square_shaped =
    indexible<T> and
    (interface::is_square_defined_for<T, b> ?
      detail::is_explicitly_square<T, b>::value :
      detail::square_shaped_impl<T, b>());
#endif


}

#endif
