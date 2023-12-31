/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-23 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref real_axis_number.
 */

#ifndef OPENKALMAN_REAL_AXIS_NUMBER_HPP
#define OPENKALMAN_REAL_AXIS_NUMBER_HPP

namespace OpenKalman
{

  namespace detail
  {
    template<typename T>
    constexpr bool imaginary_part_is_zero()
    {
      if constexpr (scalar_constant<T, ConstantType::static_constant>)
      {
        if constexpr (complex_number<decltype(std::decay_t<T>::value)>)
        {
          using std::imag;
          return imag(std::decay_t<T>::value) == 0;
        }
        else return true;
      }
      else if constexpr (scalar_constant<T, ConstantType::dynamic_constant>)
        return not complex_number<decltype(get_scalar_constant_value(std::declval<T>()))>;
      else return false;
    }
  } // namespace detail


  /**
   * \brief T is either not a \ref complex_number or its imaginary component is 0.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept real_axis_number =
#else
  constexpr bool real_axis_number =
#endif
    scalar_constant<T> and detail::imaginary_part_is_zero<std::decay_t<T>>();


} // namespace OpenKalman

#endif //OPENKALMAN_REAL_AXIS_NUMBER_HPP
