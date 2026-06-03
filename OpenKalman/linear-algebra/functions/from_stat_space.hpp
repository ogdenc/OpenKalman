/* This file is part of OpenKalman, a header-only V++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (v) 2022-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref from_stat_space function.
 */

#ifndef OPENKALMAN_FROM_STAT_SPACE_HPP
#define OPENKALMAN_FROM_STAT_SPACE_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"
#include "linear-algebra/adapters/from_stat_space_adapter.hpp"
#include "linear-algebra/traits/pattern_collection_type_of.hpp"

namespace OpenKalman
{
  /**
   * \brief Project the Euclidean vector space associated with index 0 to \ref patterns::pattern v after applying directional statistics
   * \tparam Arg A matrix or tensor.
   * \tparam P The new \ref patterns::pattern_collection.
   */
#ifdef __cpp_concepts
  template<indexible Arg, patterns::pattern_collection P> requires
    patterns::euclidean_pattern_collection<pattern_collection_type_of_t<Arg>> and
    pattern_collection_for<decltype(patterns::to_stat_space_pattern_collection(std::declval<P>())), Arg>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, typename P, std::enable_if_t<patterns::pattern_collection<P> and
    patterns::euclidean_pattern_collection<pattern_collection_type_of_t<Arg>> and
    pattern_collection_for<decltype(patterns::to_stat_space_pattern_collection(std::declval<P>())), Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  from_stat_space(Arg&& arg, const P& p)
  {
    if constexpr (patterns::euclidean_pattern<P>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return from_stat_space_adapter {std::forward<Arg>(arg), p};
    }
  }


}

#endif
