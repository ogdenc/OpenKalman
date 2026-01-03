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

#include "patterns/patterns.hpp"
#include "linear-algebra/interfaces/object_traits.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/pattern_collection_type_of.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, typename N>
    constexpr std::size_t
    numerical_N(N n)
    {
#ifdef __cpp_concepts
      if constexpr (stdex::same_as<N, values::unbounded_size_t>)
#else
      if constexpr (N == std::size_t(values::unbounded_size))
#endif
        return std::max(2_uz, index_count_v<T>);
      else
        return n;
    }


#ifndef __cpp_concepts
    template<typename T, applicability b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, applicability b>
    struct is_explicitly_square<T, b, std::enable_if_t<interface::object_traits<stdex::remove_cvref_t<T>>::template is_square<b>>>
      : std::true_type {};


    template<typename T, std::size_t N, applicability b, typename = void>
    struct same_pattern_dimensions : std::false_type {};

    template<typename T, std::size_t N, applicability b>
    struct same_pattern_dimensions<T, N, b, std::enable_if_t<
      patterns::collection_patterns_have_same_dimension<typename pattern_collection_type_of<T>::type, N, b>
      : std::true_type {};
#endif
  }


  /**
   * \brief Specifies that an object is square, meaning that the first N indices have the same extent.
   * \details N must be at least 2.
   * If N is greater than the index count, the extents of T will effectively be padded with extent 1
   * so that there are N extents.
   * If N == values::unbounded_size, N is treated as std::max(2UZ, index_count_v<T>).
   * \note A 0-by-0 array is considered to be square, but a 0-by-1 or 1-by-0 array is not.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == applicability::guaranteed</code>: T is known at compile time to be square;
   * - if <code>b == applicability::permitted</code>: It is known at compile time that T <em>may</em> be square.
   * \sa patterns::collection_patterns_have_same_dimension
   */
#ifdef __cpp_concepts
  template<typename T, auto N = values::unbounded_size, applicability b = applicability::guaranteed>
  concept square_shaped =
    indexible<T> and
    (values::size<decltype(N)> or values::integral<decltype(N)>) and
    (detail::numerical_N<T>(N) >= 2) and
    (not interface::is_square_defined_for<T, b> or
      interface::object_traits<stdex::remove_cvref_t<T>>::template is_square<b>) and
    (interface::is_square_defined_for<T, b> or
      patterns::collection_patterns_have_same_dimension<pattern_collection_type_of_t<T>, detail::numerical_N<T>(N), b>);
#else
  template<typename T, std::size_t N = values::unbounded_size, applicability b = applicability::guaranteed>
  constexpr bool square_shaped =
    indexible<T> and
    (detail::numerical_N<T>(N) >= 2) and
    (interface::is_square_defined_for<T, b> ?
      detail::is_explicitly_square<T, b>::value :
      detail::same_pattern_dimensions<T, detail::numerical_N<T, N>(), b>);
#endif


}

#endif
