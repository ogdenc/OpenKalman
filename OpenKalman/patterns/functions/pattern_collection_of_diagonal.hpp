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
 * \internal
 * \brief Definition for \ref pattern_collection_of_diagonal.
 */

#ifndef OPENKALMAN_PATTERN_COLLECTION_OF_DIAGONAL_HPP
#define OPENKALMAN_PATTERN_COLLECTION_OF_DIAGONAL_HPP

#include "collections/collections.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Any.hpp"
#include "patterns/concepts/compares_with.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/functions/get_pattern.hpp"
#include "patterns/traits/pattern_collection_element.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief Convert one \ref pattern_collection to another corresponding to the \ref diagonal_matrix of the argument.
   * \details In the result, the pattern for rank 0 will be the the pattern for rank 0 in the argument,
   * except that it is potentially truncated if the argument's pattern for rank 1 is shorter.
   */
#ifdef __cpp_concepts
  template<pattern_collection P> requires
    compares_with<
      pattern_collection_element_t<0, P>,
      pattern_collection_element_t<1, P>,
      &stdex::is_lteq, applicability::permitted> or
    compares_with<
      pattern_collection_element_t<1, P>,
      pattern_collection_element_t<0, P>,
      &stdex::is_lteq, applicability::permitted>
  constexpr pattern_collection auto
#else
  template<typename P, std::enable_if_t<
    pattern_collection<P> and
    (compares_with<
      pattern_collection_element_t<0, P>,
      pattern_collection_element_t<1, P>,
      &stdex::is_lteq, applicability::permitted> or
    compares_with<
      pattern_collection_element_t<1, P>,
      pattern_collection_element_t<0, P>,
      &stdex::is_lteq, applicability::permitted>), int> = 0>
  constexpr auto
#endif
  pattern_collection_of_diagonal(P&& p)
  {
    using N0 = std::integral_constant<std::size_t, 0>;
    using N1 = std::integral_constant<std::size_t, 1>;
    using N2 = std::integral_constant<std::size_t, 2>;

    using P0 = pattern_collection_element_t<0, P>;
    using P1 = pattern_collection_element_t<1, P>;

    if constexpr (values::fixed_value_compares_with<collections::size_of<P>, 0>)
    {
      return std::forward<P>(p);
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<P>, 1>)
    {
      return pattern_collection_of_diagonal(collections::views::concat(std::forward<P>(p), std::array{Dimensions<1>{}}));
    }
    else if constexpr (compares_with<P0, P1, &stdex::is_lteq>)
    {
      auto p0 = std::array{get_pattern(p, N0{})};
      auto ps = std::forward<P>(p) | collections::views::slice(N2{});
      return collections::views::concat(std::move(p0), std::move(ps));
    }
    else if constexpr (compares_with<P0, P1, &stdex::is_gt>)
    {
      auto p1 = std::array{get_pattern(p, N1{})};
      auto ps = std::forward<P>(p) | collections::views::slice(N2{});
      return collections::views::concat(std::move(p1), std::move(ps));
    }
    else
    {
      auto p0 = get_pattern(p, N0{});
      auto p1 = get_pattern(p, N1{});
      auto ps = std::forward<P>(p) | collections::views::slice(N2{});

      if (compare<&stdex::is_lteq>(p0, p1))
        return collections::views::concat(std::array{Any{std::move(p0)}}, std::move(ps));
      if (compare<&stdex::is_gt>(p0, p1))
        return collections::views::concat(std::array{Any{std::move(p1)}}, std::move(ps));
      throw (std::logic_error("Patterns for the first two ranks are not compatible for taking diagonal"));
    }
  }

}

#endif
