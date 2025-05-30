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
 * \brief Definition for \ref get_hash_code.
 */

#ifndef OPENKALMAN_GET_HASH_CODE_HPP
#define OPENKALMAN_GET_HASH_CODE_HPP

#include <typeindex>
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"

namespace OpenKalman::coordinates::internal
{
  /**
   * \brief Obtain a unique hash code for an \ref coordinates::descriptor.
   * \details Two coordinates will be equivalent if they have the same hash code.
   */
#ifdef __cpp_concepts
  template<descriptor Arg>
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
#endif
  constexpr std::size_t
  get_hash_code(const Arg& arg)
  {
    return interface::coordinate_descriptor_traits<std::decay_t<Arg>>::hash_code(arg);
  }


} // namespace OpenKalman::coordinates::internal


#endif //OPENKALMAN_GET_HASH_CODE_HPP
