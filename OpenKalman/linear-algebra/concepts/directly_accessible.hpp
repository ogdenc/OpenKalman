/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref directly_accessible.
 */

#ifndef OPENKALMAN_DIRECTLY_ACCESSIBLE_HPP
#define OPENKALMAN_DIRECTLY_ACCESSIBLE_HPP


namespace OpenKalman
{
  /**
   * \brief The underlying raw data for T is directly accessible.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept directly_accessible =
#else
  constexpr bool directly_accessible =
#endif
    indexible<T> and interface::raw_data_defined_for<std::decay_t<T>&>;


} // namespace OpenKalman

#endif //OPENKALMAN_DIRECTLY_ACCESSIBLE_HPP
