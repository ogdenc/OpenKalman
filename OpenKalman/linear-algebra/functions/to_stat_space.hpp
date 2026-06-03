/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief definitions for \ref to_stat_space function.
 */

#ifndef OPENKALMAN_TO_STAT_SPACE_HPP
#define OPENKALMAN_TO_STAT_SPACE_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"
#include "linear-algebra/adapters/to_stat_space_adapter.hpp"

namespace OpenKalman
{
  /**
   * \brief Project the vector space associated with index 0 to a Euclidean space for applying directional statistics.
   * \tparam Arg A matrix or tensor.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  to_stat_space(Arg&& arg)
  {
    if constexpr (patterns::euclidean_pattern<decltype(get_index_pattern<0>(std::declval<Arg&&>()))>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return to_stat_space_adapter {std::forward<Arg>(arg)};
    }
  }


}

#endif
