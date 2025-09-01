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
 * \brief Definition for \ref get_descriptor_dimension.
 */

#ifndef OPENKALMAN_GET_DESCRIPTOR_SIZE_HPP
#define OPENKALMAN_GET_DESCRIPTOR_SIZE_HPP

#include "values/values.hpp"
#include "coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "coordinates/concepts/descriptor.hpp"
#include "coordinates/concepts/euclidean_pattern.hpp"

namespace OpenKalman::coordinates::internal
{
  /**
   * \internal
   * \brief Get the size of \ref coordinates::descriptor Arg
   */
#ifdef __cpp_concepts
  template<descriptor Arg>
  constexpr values::index decltype(auto)
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  get_descriptor_dimension(Arg&& arg)
  {
    if constexpr (values::index<Arg>)
    {
      return values::cast_to<std::size_t>(std::forward<Arg>(arg));
    }
    else
    {
      using U = std::decay_t<stdcompat::unwrap_ref_decay_t<Arg>>;
      using Traits = interface::coordinate_descriptor_traits<U>;
      if constexpr (std::is_same_v<U, std::decay_t<Arg>>) return Traits::dimension(std::forward<Arg>(arg));
      else return Traits::dimension(std::forward<Arg>(arg).get());
    }
  }


}


#endif
