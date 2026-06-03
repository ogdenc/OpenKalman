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
#include "linear-algebra/concepts/dimension_size_of_index_is.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t N, applicability b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, std::size_t N, applicability b>
    struct is_explicitly_square<T, N, b, std::enable_if_t<
      interface::object_traits<stdex::remove_cvref_t<T>>::template is_square<N, b>>>
      : std::true_type {};


    template<typename T, auto N, applicability b, typename = void>
    struct same_pattern_dimensions : std::false_type {};

    template<typename T, auto N, applicability b>
    struct same_pattern_dimensions<T, N, b, std::enable_if_t<
      patterns::collection_patterns_have_same_dimension<typename pattern_collection_type_of<T>::type, N, b>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief The first 2 and up to N indices have the same extent.
   * \details If N is greater than the index count, the extents of T will effectively be padded with extent 1
   * so that there are N extents. If N == 1, T must be \ref one-dimensional.
   * \note A 0-by-0 array is considered to be square, but a 0-by-1 or 1-by-0 array is not.
   * \tparam b Defines what happens when one or more of the indices has dynamic dimension:
   * - if <code>b == applicability::guaranteed</code>: T is known at compile time to be square;
   * - if <code>b == applicability::permitted</code>: It is known at compile time that T <em>may</em> be square.
   * \sa is_square_shaped, patterns::collection_patterns_have_same_dimension
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N = 2, applicability b = applicability::guaranteed>
  concept square_shaped = indexible<T> and (N > 0) and
    ((N == 1 and dimension_size_of_index_is<T, 0, 1, &stdex::is_eq, b>) or
      (interface::is_square_defined_for<T, N, b> and
        interface::object_traits<std::remove_cvref_t<T>>::template is_square<N, b>) or
      (N != 1 and not interface::is_square_defined_for<T, N, b> and
        patterns::collection_patterns_have_same_dimension<pattern_collection_type_of_t<T>, N, b>));
#else
  template<typename T, std::size_t N = 2, applicability b = applicability::guaranteed>
  constexpr bool square_shaped = indexible<T> and (N > 0) and
    ((N == 1 and dimension_size_of_index_is<T, 0, 1, &stdex::is_eq, b>) or
      (interface::is_square_defined_for<T, N, b> and detail::is_explicitly_square<T, b>::value) or
      (N != 1 and not interface::is_square_defined_for<T, N, b> and detail::same_pattern_dimensions<T, N, b>::value));
#endif


}

#endif
