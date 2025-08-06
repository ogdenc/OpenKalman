/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref is_uniform_pattern_component_of
 */

#ifndef OPENKALMAN_IS_UNIFORM_COMPONENT_OF_HPP
#define OPENKALMAN_IS_UNIFORM_COMPONENT_OF_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/traits/uniform_pattern_type.hpp"
#include "linear-algebra/coordinates/concepts/uniform_pattern.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \internal
   * \brief Whether <code>a</code> is a 1D \ref coordinates::pattern object that, when replicated some number of times, becomes <code>c</code>.
   */
#ifdef __cpp_concepts
  template<pattern A, pattern B>
#else
  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
#endif
  constexpr auto is_uniform_pattern_component_of(const A& a, const B& b)
  {
    if constexpr (descriptor_collection<A> and not collections::sized<A>)
    {
      return std::false_type {};
    }
    else if constexpr (dimension_of_v<A> != dynamic_size and dimension_of_v<A> != 1)
    {
      return std::false_type {};
    }
    else if constexpr (fixed_pattern<A> and fixed_pattern<B> and euclidean_pattern<A> != euclidean_pattern<B>)
    {
      return std::false_type {};
    }
    else if constexpr (euclidean_pattern<B>)
    {
      return values::operation(
        std::logical_and{},
        get_is_euclidean(a),
        values::operation(
          std::logical_and{},
          values::operation(std::equal_to{}, get_dimension(a), std::integral_constant<std::size_t, 1>{}),
          [](const B& b)
          {
            if constexpr (not collections::sized<B>) return std::true_type {};
            else return values::operation(std::greater{}, get_dimension(b), std::integral_constant<std::size_t, 0>{});
          }(b)
        ));
    }
    else if constexpr (descriptor<B>)
    {
      if constexpr (fixed_pattern<A> and fixed_pattern<B>)
        return std::bool_constant<
          dimension_of_v<A> == 1 and
          compares_with<A, uniform_pattern_type_t<B>, &stdcompat::is_lteq>>{};
      else
        return get_dimension(a) == 1 and stdcompat::is_lteq(compare(a, b));
    }
    else if constexpr (uniform_pattern<B>)
    {
      if constexpr (fixed_pattern<A>)
        return std::bool_constant<compares_with<A, uniform_pattern_type_t<B>>>{};
      else
        return stdcompat::is_eq(compare(a, uniform_pattern_type_t<B>{}));
    }
    else if constexpr (collections::sized<B>)
    {
      if (get_dimension(a) != 1 or get_is_euclidean(a) != get_is_euclidean(b)) return false;
      if (get_is_euclidean(b)) return get_dimension(b) > 0;
      auto dim_b = get_dimension(b);
      if constexpr (descriptor<A>)
        return stdcompat::is_eq(compare(collections::views::repeat(a, dim_b), b));
      else
        return stdcompat::is_eq(compare(a | collections::views::all | collections::views::replicate(dim_b), b));
    }
    else
    {
      return false;
    }
  }

}

#endif
