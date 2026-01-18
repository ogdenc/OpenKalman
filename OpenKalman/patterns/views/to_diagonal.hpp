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
 * \brief Definition for \ref patterns::views::to_diagonal.
 */

#ifndef OPENKALMAN_PATTERNS_TO_DIAGONAL_HPP
#define OPENKALMAN_PATTERNS_TO_DIAGONAL_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/functions/get_pattern.hpp"

namespace OpenKalman::patterns::views
{
  namespace detail
  {
    /**
     * \internal
     * \brief A closure for the to_diagonal view.
     */
    struct to_diagonal_closure : stdex::ranges::range_adaptor_closure<to_diagonal_closure>
    {
      constexpr to_diagonal_closure() = default;

#ifdef __cpp_concepts
      template<pattern_collection R>
      constexpr pattern_collection decltype(auto)
#else
      template<typename R, std::enable_if_t<pattern_collection<R>, int> = 0>
      constexpr decltype(auto)
#endif
      operator() (R&& r) const
      {
      if constexpr (values::fixed_value_compares_with<collections::size_of<R>, 0>)
        return std::forward<R>(r);
      else
        return collections::views::concat(std::array{get_pattern<0>(r)}, std::forward<R>(r));
      }
    };


    struct to_diagonal_adapter
    {
      constexpr auto
      operator() () const
      {
        return to_diagonal_closure {};
      }


#ifdef __cpp_concepts
      template<pattern_collection R>
      constexpr pattern_collection decltype(auto)
#else
      template<typename R, std::enable_if_t<pattern_collection<R>, int> = 0>
      constexpr decltype(auto)
#endif
      operator() (R&& r) const
      {
        return to_diagonal_closure{}(std::forward<R>(r));
      }
    };
  }


  /**
   * \brief A RangeAdapterObject that converts one \ref pattern_collection to another that is equivalent to duplicating the first index.
   * \details In the result, the pattern for ranks 0 and 1 will both be the pattern for rank 0 in the argument.
   */
  inline constexpr detail::to_diagonal_adapter to_diagonal;


}

#endif
