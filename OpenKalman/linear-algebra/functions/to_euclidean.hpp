/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief definitions for \ref to_euclidean function.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_HPP
#define OPENKALMAN_TO_EUCLIDEAN_HPP

#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/vector_space_descriptor_of.hpp"
#include "linear-algebra/adapters/ToEuclideanExpr.hpp"
#include "linear-algebra/interfaces/library-interfaces-defined.hpp"

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
  to_euclidean(Arg&& arg)
  {
    if constexpr (euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::to_euclidean_defined_for<Arg, Arg&&>)
    {
      return interface::library_interface<std::decay_t<Arg>>::to_euclidean(std::forward<Arg>(arg));
    }
    else
    {
      return ToEuclideanExpr {std::forward<Arg>(arg)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_EUCLIDEAN_HPP
