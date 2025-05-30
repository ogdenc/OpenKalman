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
#ifdef __cpp_lib_concepts
  namespace detail
  {
    template<typename T>
    concept raw_data_result = requires(T t) { {*t} -> values::scalar; };
  }
#endif


  /**
   * \internal
   * \brief Returns a pointer to the raw data of a directly accessible tensor or matrix.
   */
#ifdef __cpp_concepts
  template<interface::raw_data_defined_for Arg>
  constexpr detail::raw_data_result decltype(auto) raw_data(Arg&& arg)
#else
  template<typename Arg, std::enable_if_t<interface::raw_data_defined_for<Arg>, int> = 0>
  constexpr decltype(auto) raw_data(Arg&& arg)
#endif
  {
    return interface::indexible_object_traits<std::decay_t<Arg>>::raw_data(std::forward<Arg>(arg));
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_RAW_DATA_HPP
