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
 * \brief Definition for \ref coordinate::get_component_count.
 */

#ifndef OPENKALMAN_GET_COMPONENT_COUNT_HPP
#define OPENKALMAN_GET_COMPONENT_COUNT_HPP

#include "basics/functions/get_collection_size.hpp"
#include "values/concepts/index.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief Get the number of components of a \ref coordinate::pattern
   */
#ifdef __cpp_concepts
  template<pattern T>
constexpr value::index auto
#else
  template<typename T, std::enable_if_t<pattern<T>, int> = 0>
  constexpr auto
#endif
  get_component_count(const T& t)
  {
    if constexpr (descriptor<T>)
    {
      return std::integral_constant<std::size_t, 1_uz>{};
    }
    else
    {
      return get_collection_size(t);
    }

  }


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_GET_COMPONENT_COUNT_HPP
