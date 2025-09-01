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
#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/uniform_pattern.hpp"
#include "coordinates/concepts/compares_with.hpp"
#include "coordinates/traits/uniform_pattern_type.hpp"
#include "coordinates/concepts/uniform_pattern.hpp"
#include "coordinates/functions/compare.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \internal
   * \brief Determine whether a is a uniform pattern component of b.
   * \details A uniform pattern component is a 1D component of a \ref coordinates::pattern "pattern" that,
   * when replicated some unspecified number of times n >= 0, is equivalent to the pattern.
   */
#ifdef __cpp_concepts
  template<pattern A, pattern B>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr auto
#endif
  is_uniform_pattern_component_of(const A& a, const B& b)
  {
    if constexpr (not uniform_pattern<B, applicability::permitted>)
    {
      return std::false_type {};
    }
    else if constexpr (euclidean_pattern<B>)
    {
      return compare(a, Dimensions<1>{});;
    }
    else
    {
      using C = common_descriptor_type_t<B>;
      if constexpr (compares_with<A, C, &stdcompat::is_eq, applicability::guaranteed>)
      {
        return values::operation(std::equal_to{}, get_dimension(a), std::integral_constant<std::size_t, 1>{});
      }
      else if constexpr (compares_with<A, C, &stdcompat::is_neq, applicability::guaranteed> or
        (dimension_of_v<C> != dynamic_size and dimension_of_v<C> != 1))
      {
        return std::false_type {};
      }
      else if (get_dimension(a) != 1)
      {
        return false;
      }
      else if (get_is_euclidean(b))
      {
        return get_is_euclidean(a);
      }
      else
      {
        if constexpr (descriptor<B>)
        {
          return compare(a, b);
        }
        else
        {
          for (const auto& x : collections::views::all(b))
          {
            if (get_dimension(x) == 0) continue;
            if (get_is_euclidean(x) and get_is_euclidean(a)) continue;
            if (not compare(a, x)) return false;
          }
          return true;
        }
      }
    }
  }

}

#endif
