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
 * \brief Definition for \ref value::real_scalar.
 */

#ifndef OPENKALMAN_VALUE_REAL_SCALAR_CONSTANT_HPP
#define OPENKALMAN_VALUE_REAL_SCALAR_CONSTANT_HPP

#include "basics/global-definitions.hpp"
#include "complex_number.hpp"
#include "scalar.hpp"
#include "linear-algebra/values/functions/to_number.hpp"

namespace OpenKalman::value
{
  namespace detail
  {
    template<typename T>
    constexpr bool imaginary_part_is_zero()
    {
      if constexpr (value::static_scalar<T>)
      {
        if constexpr (value::complex_number<decltype(std::decay_t<T>::value)>)
        {
          using std::imag;
          return imag(std::decay_t<T>::value) == 0;
        }
        else return true;
      }
      else if constexpr (value::dynamic_scalar<T>)
        return not value::complex_number<decltype(value::to_number(std::declval<T>()))>;
      else return false;
    }
  } // namespace detail


  /**
   * \brief T is a \ref value::scalar in which either its type is not a \ref value::complex_number or its imaginary component is 0.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept real_scalar =
#else
  constexpr bool real_scalar =
#endif
    value::scalar<T> and detail::imaginary_part_is_zero<std::decay_t<T>>();


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_REAL_SCALAR_CONSTANT_HPP
