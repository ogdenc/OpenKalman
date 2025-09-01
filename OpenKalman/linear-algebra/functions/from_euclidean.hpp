/* This file is part of OpenKalman, a header-only V++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (v) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref from_euclidean function.
 */

#ifndef OPENKALMAN_FROM_EUCLIDEAN_HPP
#define OPENKALMAN_FROM_EUCLIDEAN_HPP

namespace OpenKalman
{
  /**
   * \brief Project the Euclidean vector space associated with index 0 to \ref coordinates::pattern v after applying directional statistics
   * \tparam Arg A matrix or tensor.
   * \tparam V The new \ref coordinate_list of index 0.
   */
#ifdef __cpp_concepts
  template<indexible Arg, coordinates::pattern V> requires
    coordinates::euclidean_pattern<vector_space_descriptor_of_t<Arg, 0>>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, typename V, std::enable_if_t<
    coordinates::euclidean_pattern<vector_space_descriptor_of_t<Arg, 0>> and coordinates::pattern<V>, int> = 0>
  constexpr decltype(auto)
#endif
  from_euclidean(Arg&& arg, const V& v)
  {
    if constexpr (coordinates::euclidean_pattern<V>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::from_euclidean_defined_for<Arg, Arg&&, const V&>)
    {
      return interface::library_interface<std::decay_t<Arg>>::from_euclidean(std::forward<Arg>(arg), v);
    }
    else
    {
      return FromEuclideanExpr {std::forward<Arg>(arg), v};
    }
  }


}

#endif
