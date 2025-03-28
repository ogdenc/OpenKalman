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
 * \brief Definition for \ref get_descriptor_size.
 */

#ifndef OPENKALMAN_GET_DESCRIPTOR_SIZE_HPP
#define OPENKALMAN_GET_DESCRIPTOR_SIZE_HPP

#include "values/concepts/index.hpp"
#include "values/functions/cast_to.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"

namespace OpenKalman::coordinate::internal
{
  /**
   * \internal
   * \brief Get the size of \ref coordinate::descriptor Arg
   */
#ifdef __cpp_concepts
  template<descriptor Arg>
  constexpr value::index decltype(auto)
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  get_descriptor_size(Arg&& arg)
  {
    if constexpr (interface::coordinate_descriptor_traits<std::decay_t<Arg>>::is_specialized)
    {
      return interface::coordinate_descriptor_traits<std::decay_t<Arg>>::size(std::forward<Arg>(arg));
    }
    else
    {
      static_assert(value::index<Arg>);
      return value::cast_to<std::size_t>(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman::coordinate::internal


#endif //OPENKALMAN_GET_DESCRIPTOR_SIZE_HPP
