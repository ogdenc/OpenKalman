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
 * \brief Definition for \ref square_shaped.
 */

#ifndef OPENKALMAN_SQUARE_SHAPED_HPP
#define OPENKALMAN_SQUARE_SHAPED_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/interfaces/object_traits.hpp"
#include "linear-algebra/traits/pattern_collection_type_of.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<auto N, typename T>
    constexpr std::size_t
    N_adjusted()
    {
#ifdef __cpp_concepts
      if constexpr (std::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>)
#else
      if constexpr (N == std::size_t(values::unbounded_size))
#endif
      {
        if constexpr (values::fixed<index_count<T>>)
          return std::max(2_uz, index_count_v<T>);
        else
          return 2_uz;
      }
      else return N;
    }


#ifndef __cpp_concepts
    template<typename T, applicability b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, applicability b>
    struct is_explicitly_square<T, b, std::enable_if_t<
      interface::object_traits<stdex::remove_cvref_t<T>>::template is_square<b>>>
      : std::true_type {};


    template<typename T, auto N, applicability b, typename = void>
    struct same_pattern_dimensions : std::false_type {};

    template<typename T, auto N, applicability b>
    struct same_pattern_dimensions<T, N, b, std::enable_if_t<
      patterns::collection_patterns_have_same_dimension<typename pattern_collection_type_of<T>::type, N_adjusted<N, T>(), b>>>
      : std::true_type {};
#endif
  }


  /**
   * \brief At least 2 and at most N indices have the same extent.
   * \details N must be at least 2 or must be values::unbounded_size.
   * If the latter, at least two indicess will be compared.
   * If N is greater than the index count, the extents of T will effectively be padded with extent 1
   * so that there are N extents.
   * \note A 0-by-0 array is considered to be square, but a 0-by-1 or 1-by-0 array is not.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == applicability::guaranteed</code>: T is known at compile time to be square;
   * - if <code>b == applicability::permitted</code>: It is known at compile time that T <em>may</em> be square.
   * \sa is_square_shaped, patterns::collection_patterns_have_same_dimension
   */
#ifdef __cpp_concepts
  template<typename T, auto N = values::unbounded_size, applicability b = applicability::guaranteed>
  concept square_shaped =
    indexible<T> and
    (values::integral<decltype(N)> or std::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N >= 2) and
    (not interface::is_square_defined_for<T, b> or
      interface::object_traits<std::remove_cvref_t<T>>::template is_square<b>) and
    (interface::is_square_defined_for<T, b> or
      patterns::collection_patterns_have_same_dimension<pattern_collection_type_of_t<T>, detail::N_adjusted<N, T>(), b>);
#else
  template<typename T, std::size_t N = values::unbounded_size, applicability b = applicability::guaranteed>
  constexpr bool square_shaped =
    indexible<T> and
    (N == values::unbounded_size or N >= 2) and
    (interface::is_square_defined_for<T, b> ?
      detail::is_explicitly_square<T, b>::value :
      detail::same_pattern_dimensions<T, N, b>::value);
#endif


}

#endif
