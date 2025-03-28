/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref get_descruptor_euclidean_size.
 */

#ifndef OPENKALMAN_GET_DESCRIPTOR_EUCLIDEAN_SIZE_HPP
#define OPENKALMAN_GET_DESCRIPTOR_EUCLIDEAN_SIZE_HPP

#include "values/concepts/index.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"

namespace OpenKalman::coordinate::internal
{
  /**
   * \internal
   * \brief Get the euclidean size of \ref coordinate::descriptor Arg
   * \details This is the number of coordinates when a corresponding vector is transformed to Euclidan space for direcitonal statistics.
   */
#ifdef __cpp_concepts
  template<descriptor Arg>
  constexpr value::index decltype(auto)
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  get_descriptor_euclidean_size(Arg&& arg)
  {
    if constexpr (interface::coordinate_descriptor_traits<std::decay_t<Arg>>::is_specialized)
    {
      return interface::coordinate_descriptor_traits<std::decay_t<Arg>>::euclidean_size(std::forward<Arg>(arg));
    }
    else
    {
      static_assert(value::index<Arg>);
      return std::forward<Arg>(arg);
    }
  }


} // namespace OpenKalman::coordinate::internal


#endif //OPENKALMAN_GET_DESCRIPTOR_EUCLIDEAN_SIZE_HPP
