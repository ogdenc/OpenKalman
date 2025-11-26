/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref most_fixed_pattern.
 */

#ifndef OPENKALMAN_MOST_FIXED_PATTERN_HPP
#define OPENKALMAN_MOST_FIXED_PATTERN_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/compares_with.hpp"

namespace OpenKalman::coordinates::internal
{
  namespace detail
  {
    class MostFixedPattern
    {
      template<typename D, typename...Ds>
      constexpr decltype(auto)
      impl(D&& d, Ds&&...ds)
      {
        if constexpr (sizeof...(Ds) == 0) return std::forward<D>(d);
        else if constexpr (fixed_pattern<D>) return std::forward<D>(d);
        else return impl(std::forward<Ds>(ds)...);
      }

    public:

      template<typename D, typename...Ds>
      constexpr decltype(auto)
      operator()(D&& d, Ds&&...ds)
      {
        static_assert((... and compares_with<D, Ds, &stdex::is_eq, applicability::permitted>),
          "In most_fixed_pattern, elements of pattern_collection argument are not compatible");
        return impl(std::forward<D>(d), std::forward<Ds>(ds)...);
      }
    };
  }


  /**
   * \brief Given a fixed-size /ref coordinates::pattern_collection, return the first component, if any, that is a \ref fixed_pattern.
   * \details If there are no fixed patterns, returns the last pattern.
   */
#ifdef __cpp_concepts
  template<pattern_collection P> requires values::fixed_value_compares_with<collections::size_of<P>, 0, &std::is_gt>
  constexpr pattern decltype(auto)
#else
  template<typename P, std::enable_if_t<values::fixed_value_compares_with<collections::size_of<P>, 0, &stdex::is_gt>, int> = 0>
  constexpr decltype(auto)
#endif
  most_fixed_pattern(P&& p)
  {
    return collections::apply(detail::MostFixedPattern{}, std::forward<P>(p));
  }


}

#endif
