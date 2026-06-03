/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref attach_patterns and \ref pattern_adapter.
 */

#ifndef OPENKALMAN_ATTACH_PATTERNS_HPP
#define OPENKALMAN_ATTACH_PATTERNS_HPP

#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/adapters/pattern_adapter.hpp"

namespace OpenKalman
{
  namespace detail
  {
    /**
     * \internal
     * \brief Tests whether the argument is a specialization of \ref pattern_adapter.
     */
    template <typename T>
    struct is_pattern_adapter : std::false_type {};

    template <typename N, typename P>
    struct is_pattern_adapter<pattern_adapter<N, P>> : std::true_type {};

    /**
     * \internal
     * \brief Helper template for \ref is_pattern_adapter.
     */
    template<typename T>
    constexpr auto is_pattern_adapter_v = is_pattern_adapter<T>::value;

  }


  /**
   * \brief Attach a \ref patterns::pattern_collection "pattern_collection" to an \ref indexible object.
   * \details Any pattern_collection already associated with the argument will be overwritten.
   * Does not verify that any dynamic extents match with the attached pattern.
   * \sa pattern_adapter
   */
#ifdef __cpp_concepts
  template<indexible Arg, pattern_collection_for<Arg> P>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, typename P, std::enable_if_t<
    indexible<Arg> and pattern_collection_for<P, Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  attach_patterns(Arg&& arg, P&& p)
  {
    if constexpr (compares_with_pattern_collection<Arg, P, &stdex::is_eq, applicability::guaranteed>)
      return std::forward<Arg>(arg);
    else if constexpr (detail::is_pattern_adapter<std::decay_t<Arg>>::value)
      return attach_patterns(std::forward<Arg>(arg).nested_object(), std::forward<P>(p));
    else
      return pattern_adapter {std::forward<Arg>(arg), std::forward<P>(p)};
  }


}


#endif
