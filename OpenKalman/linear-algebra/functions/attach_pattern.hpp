/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref attach_pattern.
 */

#ifndef OPENKALMAN_MAKE_ATTACH_PATTERN_HPP
#define OPENKALMAN_MAKE_ATTACH_PATTERN_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/adapters/pattern_adapter.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template <typename>
    struct is_pattern_adapter : std::false_type {};

    template <typename N, typename P>
    struct is_pattern_adapter<pattern_adapter<N, P>> : std::true_type {};
  }


  /**
   * \brief Attach a \ref patterns::pattern_collection "pattern_collection" to an \ref indexible object.
   * \details Any pattern_collection already associated with the argument will be overwritten.
   */
#ifdef __cpp_concepts
  template<indexible Arg, pattern_collection_for<Arg> P>
#else
  template<typename Arg, typename P, std::enable_if_t<
    indexible<Arg> and pattern_collection_for<P, Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  attach_pattern(Arg&& arg, P&& p)
  {
    if constexpr (detail::is_pattern_adapter<std::decay_t<Arg>>::value)
      return attach_pattern(std::forward<Arg>(arg).nested_object(), std::forward<P>(p));
    else if constexpr (patterns::euclidean_pattern_collection<P>)
      return std::forward<Arg>(arg);
    else
      return pattern_adapter {std::forward<Arg>(arg), std::forward<P>(p)};
  }

}

#endif
