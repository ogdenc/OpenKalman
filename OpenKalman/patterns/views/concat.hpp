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
 * \brief Definition of \ref patterns::views::concat.
 */

#ifndef OPENKALMAN_PATTERNS_VIEWS_CONCAT_HPP
#define OPENKALMAN_PATTERNS_VIEWS_CONCAT_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/functions/get_dimension.hpp"

namespace OpenKalman::patterns::views
{
  namespace detail
  {
    struct concat_adaptor
    {
  #ifdef __cpp_concepts
      template<pattern...P>
  #else
      template<typename...P, std::enable_if_t<(... and pattern<P>), int> = 0>
  #endif
      constexpr auto
      operator() (P&&...p) const
      {
        if constexpr ((... and euclidean_pattern<P>))
          return Dimensions {values::operation([](auto...xs){ return (0 + ... + xs); }, get_dimension(p)...)};
        else if constexpr ((... and descriptor<P>))
          return std::make_tuple(std::forward<P>(p)...) | collections::views::all;
        else
          return collections::views::concat([](P&& p) {
            if constexpr (descriptor<P>) return std::array {std::forward<P>(p)};
            else return std::forward<P>(p);
          }(std::forward<P>(p))...);
      }
    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for a set of concatenated \ref pattern objects.
   */
  inline constexpr detail::concat_adaptor concat;

}


#endif
