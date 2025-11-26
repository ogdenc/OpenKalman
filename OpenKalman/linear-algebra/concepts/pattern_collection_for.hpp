/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref pattern_collection_for.
 */

#ifndef OPENKALMAN_COMPATIBLE_SHAPE_WITH_HPP
#define OPENKALMAN_COMPATIBLE_SHAPE_WITH_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename P, typename T, applicability a, typename = std::make_index_sequence<collections::size_of<P>::value>>
    struct compatible_shape_with_impl : std::false_type {};

    template<typename P, typename T, applicability a, std::size_t...i>
    struct compatible_shape_with_impl<P, T, a, std::index_sequence<i...>>
      : std::bool_constant<(... and
        (coordinates::dimension_of_v<collections::collection_element_t<i, P>> == index_dimension_of_v<T, i> and
          (i < index_count_v<T> or
            coordinates::compares_with<collections::collection_element_t<i, P>, coordinates::Dimensions<1>>)))> {};

#ifndef __cpp_concepts
    template<typename P, typename T, applicability a, typename = void>
    struct compatible_shape_with_impl_cpp17 : std::false_type{};

    template<typename P, typename T, applicability a>
      struct compatible_shape_with_impl_cpp17<P, T, a, std::enable_if_t<
        coordinates::pattern_collection<P> and
        indexible<T> and
        collections::size_of<P>::value != stdex::dynamic_extent>>
          : compatible_shape_with_impl<P, T, a> {};
#endif
  }


  /**
   * \brief \ref coordinates::pattern_collection "pattern collection" P is compatible with \ref indexible T.
   * \details If one or the other is dynamic, both must be dynamic, otherwise, the static dimensions must match.
   * Any trailing patterns in P must be equivalent to Dimensions<1>.
   * \tparam P a \ref coordinates::pattern_collection
   * \tparam T an \ref indexible object
   */
  template<typename P, typename T, applicability a = applicability::permitted>
#ifdef __cpp_concepts
  concept pattern_collection_for =
    coordinates::pattern_collection<P> and
    indexible<T> and
    values::fixed_value_compares_with<collections::size_of<P>, std::dynamic_extent, &std::is_neq> and
    detail::compatible_shape_with_impl<P, T, a>::value;
#else
  constexpr bool pattern_collection_for = detail::compatible_shape_with_impl_cpp17<P, T, a>::value;
#endif


}

#endif
