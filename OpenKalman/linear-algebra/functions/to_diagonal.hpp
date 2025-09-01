/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref to_diagonal function.
 */

#ifndef OPENKALMAN_TO_DIAGONAL_HPP
#define OPENKALMAN_TO_DIAGONAL_HPP

namespace OpenKalman
{
  /**
   * \brief Convert an \ref indexible object into a \ref diagonal matrix.
   * \returns A \ref diagonal matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr diagonal_matrix decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  to_diagonal(Arg&& arg)
  {
    if constexpr (one_dimensional<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::to_diagonal_defined_for<Arg, Arg&&>)
    {
      return interface::library_interface<std::decay_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
    else
    {
      return diagonal_adapter {std::forward<Arg>(arg)};
    }
  }


}

#endif
