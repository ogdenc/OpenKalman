/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file \internal
 * \brief Definition for \ref value::internal::NaN.
 */

#ifndef OPENKALMAN_VALUE_NAN_HPP
#define OPENKALMAN_VALUE_NAN_HPP

#include <limits>
#include <stdexcept>
#include "values/concepts/number.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/functions/internal/make_complex_number.hpp"

namespace OpenKalman::value::internal
{
  /**
   * \internal
   * \brief Return a NaN in type T or raise an exception if Nan is not available.
   */
#ifdef __cpp_concepts
  template <value::number T>
#else
  template <typename T, std::enable_if_t<value::number<T>, int> = 0>
#endif
  constexpr std::decay_t<T> NaN()
  {
    using R = std::decay_t<T>;
    if constexpr (value::complex<T>)
      return value::internal::make_complex_number<T>(
        value::internal::NaN<value::real_type_of_t<T>>(), value::internal::NaN<value::real_type_of_t<T>>());
    else if constexpr (std::numeric_limits<R>::has_quiet_NaN)
      return std::numeric_limits<R>::quiet_NaN();
    else if constexpr (std::numeric_limits<R>::has_signaling_NaN)
      return std::numeric_limits<R>::signaling_NaN();
    else
      throw std::domain_error {"Domain error in arithmetic operation: result is not a number"};
  }

} // namespace OpenKalman::value::internal


#endif //OPENKALMAN_VALUE_NAN_HPP
