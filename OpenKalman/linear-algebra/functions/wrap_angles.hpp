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
 * \brief Definitions for \ref wrap_angles function.
 */

#ifndef OPENKALMAN_WRAP_ANGLES_HPP
#define OPENKALMAN_WRAP_ANGLES_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  wrap_angles(Arg&& arg)
  {
    if constexpr (patterns::euclidean_pattern<vector_space_descriptor_of_t<Arg, 0>> or identity_matrix<Arg> or zero<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::wrap_angles_defined_for<Arg, Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::wrap_angles(std::forward<Arg>(arg), get_pattern_collection<0>(arg));
    }
    else
    {
      return from_euclidean(to_euclidean(std::forward<Arg>(arg)), get_pattern_collection<0>(arg));
    }
  }


}

#endif
