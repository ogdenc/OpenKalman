/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_descriptor_hash_code.
 */

#ifndef OPENKALMAN_GET_HASH_CODE_HPP
#define OPENKALMAN_GET_HASH_CODE_HPP

#include "values/concepts/index.hpp"
#include "patterns/interfaces/pattern_descriptor_traits.hpp"
#include "patterns/concepts/descriptor.hpp"

namespace OpenKalman::patterns::internal
{
  /**
   * \brief Obtain a unique hash code for an \ref patterns::descriptor.
   * \details Two coordinates will be equivalent if they have the same hash code.
   */
#ifdef __cpp_concepts
  template<descriptor Arg>
  constexpr std::convertible_to<std::size_t> auto
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
  constexpr auto
#endif
  get_descriptor_hash_code(const Arg& arg)
  {
    if constexpr (values::index<Arg>)
    {
      return values::cast_to<std::size_t>(arg);
    }
    else
    {
      using U = std::decay_t<stdex::unwrap_reference_t<Arg>>;
      using Traits = interface::pattern_descriptor_traits<U>;
      if constexpr (std::is_same_v<U, Arg>) return Traits::hash_code(arg);
      else return Traits::hash_code(arg.get());
    }
  }


}


#endif
