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
 * \brief Definition for \ref value:real_type_of_t.
 */

#ifndef OPENKALMAN_VALUES_REAL_TYPE_OF_T_HPP
#define OPENKALMAN_VALUES_REAL_TYPE_OF_T_HPP

#include <type_traits>
#include "values/concepts/value.hpp"
#include "values/functions/to_number.hpp"
#include "values/math/real.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Obtain the real type associated with a number (typically a \ref value::complex number.
   * \details This will be the type of the result of <code>value::real(...)</code>.
   */
#ifdef __cpp_concepts
  template<value T> requires requires(T t) { {value::real(t)} -> value; }
#else
  template<typename T, std::enable_if_t<value<T>, int> = 0, typename = std::void_t<decltype(value::real(std::declval<T>()))>>
#endif
  using real_type_of_t = std::decay_t<decltype(value::real(std::declval<T>()))>;


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUES_REAL_TYPE_OF_T_HPP
