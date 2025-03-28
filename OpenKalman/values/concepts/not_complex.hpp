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
 * \brief Definition for \ref value::not_complex.
 */

#ifndef OPENKALMAN_VALUE_NOT_COMPLEX_HPP
#define OPENKALMAN_VALUE_NOT_COMPLEX_HPP

#include "complex.hpp"
#include "value.hpp"
#include "values/functions/to_number.hpp"
#include "values/math/imag.hpp"

namespace OpenKalman::value
{
  namespace detail
  {
    template<typename T>
    constexpr bool imaginary_part_is_zero()
    {
      if constexpr (value::fixed<T>)
      {
        return value::imag(value::to_number(std::decay_t<T>{})) == 0;
      }
      else return false;
    }
  } // namespace detail


  /**
   * \brief T is a \ref value::value in which either its type is not a \ref value::complex or its imaginary component is 0.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept not_complex =
#else
  constexpr bool not_complex =
#endif
    value::value<T> and (not complex<T> or detail::imaginary_part_is_zero<std::decay_t<T>>());


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_NOT_COMPLEX_HPP
