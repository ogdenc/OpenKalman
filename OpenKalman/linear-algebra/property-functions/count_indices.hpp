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
 * \brief Definition of \ref count_indices.
 */

#ifndef OPENKALMAN_COUNT_INDICES_HPP
#define OPENKALMAN_COUNT_INDICES_HPP

#include "linear-algebra/interfaces/object-traits-defined.hpp"

namespace OpenKalman
{
  /**
   * \brief Get the number of indices available to address the components of an \ref indexible object.
   * \sa index_count
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T>
  constexpr values::index auto
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T>, int> = 0>
  constexpr auto
#endif
  count_indices(const T& t)
  {
    return stdcompat::invoke(interface::indexible_object_traits<T>::count_indices, t);
  }


}

#endif
