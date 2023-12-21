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
 * \internal
 * \brief Definition for raw_data function.
 */

#ifndef OPENKALMAN_RAW_DATA_HPP
#define OPENKALMAN_RAW_DATA_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Returns a pointer to the raw data of a directly accessible tensor or matrix.
   */
#ifdef __cpp_concepts
  template<interface::raw_data_defined_for T>
#else
  template<typename T, std::enable_if_t<interface::raw_data_defined_for<T>, int> = 0>
#endif
  constexpr auto * const raw_data(T&& t)
  {
    return interface::indexible_object_traits<std::decay_t<T>>::raw_data(std::forward<T>(t));
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_RAW_DATA_HPP
