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
 * \brief Definition for \ref internal::singular_component.
 */

#ifndef OPENKALMAN_GET_SINGULAR_COMPONENT_HPP
#define OPENKALMAN_GET_SINGULAR_COMPONENT_HPP

#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/access.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Get the initial component of Arg, which will be the singular component in a one-dimensional object.
   */
  template<typename Arg>
  constexpr decltype(auto)
  get_singular_component(Arg&& arg)
  {
    auto mds = get_mdspan(std::forward<Arg>(arg));
    constexpr std::size_t rank = std::decay_t<decltype(mds)>::rank();
    return mds[std::array<std::integral_constant<std::size_t, 0>, rank>{}];
  }


}

#endif
