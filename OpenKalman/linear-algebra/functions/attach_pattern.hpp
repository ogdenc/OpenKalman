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

#include "coordinates/coordinates.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/adapters/pattern_adapter.hpp"

namespace OpenKalman
{
  /**
   * \brief Attach a \ref coordinates::pattern_collection "pattern_collection" to an \ref indexible object.
   * \details Any pattern_collection already associated with the argument will be overwritten.
   */
#ifdef __cpp_concepts
  template<indexible Arg, pattern_collection_for<Arg> P>
#else
  template<typename Arg, typename P, std::enable_if_t<
    indexible<Arg> and pattern_collection_for<P, Arg>, int> = 0>
#endif
  constexpr auto attach_pattern(Arg&& arg, P&& p)
  {
    if constexpr (coordinates::euclidean_pattern_collection<P>)
      return std::forward<Arg>(arg);
    else
      return pattern_adapter{std::forward<Arg>(arg), std::forward<P>(p)};
  }


  /**
   * \overload
   * \brief The pattern_collection is constructed from a list of \ref coordinates::pattern objects.
   */
#ifdef __cpp_concepts
  template<indexible Arg, coordinates::pattern...Ps> requires pattern_collection_for<std::tuple<Ps...>, Arg>
#else
  template<typename Arg, typename...Ps, std::enable_if_t<
    indexible<Arg> and (... and coordinates::pattern<Ps>) and
    pattern_collection_for<std::tuple<Ps...>, Arg>, int> = 0>
#endif
  constexpr auto attach_pattern(Arg&& arg, Ps&&...ps)
  {
    return attach_pattern(std::forward<Arg>(arg), std::tuple {std::forward<Ps>(ps)...});
  }


  /**
   * \overload
   * \brief The pattern_collection is default initializable.
   */
#ifdef __cpp_concepts
  template<coordinates::pattern_collection P, indexible Arg> requires
    pattern_collection_for<P, Arg> and
    std::default_initializable<P>
#else
  template<typename P, typename Arg, std::enable_if_t<
    indexible<Arg> and
    pattern_collection_for<P, Arg>, int> = 0>
#endif
  constexpr auto attach_pattern(Arg&& arg)
  {
    return attach_pattern(std::forward<Arg>(arg), P{});
  }


}

#endif
