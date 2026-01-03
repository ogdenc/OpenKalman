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

#include "patterns/patterns.hpp"
#include "compares_with_pattern_collection.hpp"

namespace OpenKalman
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename P, typename T, applicability a, typename = void>
    struct pattern_collection_for_impl : std::false_type{};

    template<typename P, typename T, applicability a>
      struct pattern_collection_for_impl<P, T, a, std::enable_if_t<
        patterns::collection_compares_with<
          decltype(patterns::to_extents(std::declval<P>())),
          decltype(get_pattern_collection(get_mdspan(std::declval<std::add_lvalue_reference_t<T>>()))),
          &stdex::is_eq, a>>>
          : compatible_shape_with_impl<P, T, a> {};
#endif
  }


  /**
   * \brief \ref patterns::pattern_collection "pattern collection" P has a shape that is attachable to \ref indexible T.
   * \tparam P a \ref patterns::pattern_collection
   * \tparam T an \ref indexible object
   * \sa attach_pattern
   */
  template<typename P, typename T, applicability a = applicability::permitted>
#ifdef __cpp_concepts
  concept pattern_collection_for =
    patterns::collection_compares_with<
      decltype(patterns::to_extents(std::declval<P>())),
      decltype(get_pattern_collection(get_mdspan(std::declval<std::add_lvalue_reference_t<T>>()))),
      &stdex::is_eq, a>;
#else
  constexpr bool pattern_collection_for = pattern_collection_for_impl<P, T, a>::value;
#endif


}

#endif
