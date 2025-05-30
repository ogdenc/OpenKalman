/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value:number_type_of_t.
 */

#ifndef OPENKALMAN_VALUES_NUMBER_TYPE_OF_T_HPP
#define OPENKALMAN_VALUES_NUMBER_TYPE_OF_T_HPP

#include <type_traits>
#include "values/concepts/value.hpp"
#include "values/functions/to_number.hpp"


namespace OpenKalman::values
{
  /**
   * \brief Obtain the \ref values::number type associated with a\ref values::value.
   */
#ifdef __cpp_concepts
  template<values::value T>
#else
  template<typename T, std::enable_if_t<values::value<T>, int> = 0>
#endif
  using number_type_of_t = std::decay_t<decltype(values::to_number(std::declval<T>()))>;


} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUES_NUMBER_TYPE_OF_T_HPP
