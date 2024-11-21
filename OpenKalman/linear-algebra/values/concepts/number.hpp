/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value::number.
 */

#ifndef OPENKALMAN_VALUE_SCALAR_HPP
#define OPENKALMAN_VALUE_SCALAR_HPP

#include <limits>
#include <type_traits>
#include "complex_number.hpp"

namespace OpenKalman::value
{
  /**
   * \brief T is a numerical type.
   * \details T can be any arithmetic, complex, or custom number type in which certain traits in
   * interface::number_traits are defined and typical math operations (+, -, *, /, and ==) are also defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept number =
#else
  constexpr bool number =
#endif
    std::numeric_limits<std::decay_t<T>>::is_specialized or value::complex_number<T>;


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_SCALAR_HPP
