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
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/functions/to_number.hpp"


namespace OpenKalman::value
{
  /**
   * \brief Obtain the \ref value::number type associated with a\ref value::value.
   */
#ifdef __cpp_concepts
  template<value::value T>
#else
  template<typename T, std::enable_if_t<value::value<T>, int> = 0>
#endif
  using number_type_of_t = std::decay_t<decltype(value::to_number(std::declval<T>()))>;


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUES_NUMBER_TYPE_OF_T_HPP
