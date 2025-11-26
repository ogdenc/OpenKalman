/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns_match_with.
 */

#ifndef OPENKALMAN_PATTERNS_MATCH_WITH_HPP
#define OPENKALMAN_PATTERNS_MATCH_WITH_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename...Ps>
    struct patterns_match_with_impl : std::true_type {};

    template<typename P, typename...Ps>
    struct patterns_match_with_impl<P, Ps...>
      : std::bool_constant<(... and (coordinates::pattern_collection_compares_with<P, Ps>))> {};

#ifndef __cpp_concepts
    template<typename = void, typename...Ts>
    struct pattern_mchx : std::false_type {};

    template<typename...Ts>
    struct pattern_mchx<std::enable_if_t<
      patterns_match_with_impl<decltype(get_pattern_collection(std::declval<Ts>()))...>::value>, Ts...>
      : std::true_type {};
#endif
  }


  /**
   * \brief Specifies that a set of indexible objects have equivalent vector space descriptors for each index.
   * \tparam Ts A set of \ref indexible objects
   * \sa patterns_may_match_with
   * \sa patterns_match
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept patterns_match_with =
    (indexible<Ts> and ...) and
    detail::patterns_match_with_impl<decltype(get_pattern_collection(std::declval<Ts>()))...>::value;
#else
  constexpr bool patterns_match_with =
    (indexible<Ts> and ...) and
    detail::pattern_mchx<void, Ts...>::value;
#endif

}

#endif
