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
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/functions/to_number.hpp"


namespace OpenKalman::value
{
  /**
   * \brief Obtain the real type associated with a number (typically a \ref value::complex number.
   */
#ifdef __cpp_concepts
  template<value::value T>
#else
  template<typename T, std::enable_if_t<value::value<T>, int> = 0>
#endif
  using real_type_of_t = typename interface::number_traits<std::decay_t<T>>::real_type;


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUES_REAL_TYPE_OF_T_HPP
