/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref value::fixed constants.
 */

#ifndef OPENKALMAN_VALUE_CLASSES_FIXED_CONSTANTS_HPP
#define OPENKALMAN_VALUE_CLASSES_FIXED_CONSTANTS_HPP

#include "values/concepts/number.hpp"
#include "values/classes/Fixed.hpp"

namespace OpenKalman::value
{
  /**
   * \brief A fixed version of pi
   */
#if __cpp_nontype_template_args >= 201911L
  template<value::number T>
  using fixed_pi = value::Fixed<T, numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_pi
  {
    static_assert(value::number<T>);
    using value_type = T;
    static constexpr value_type value {numbers::pi_v<value_type>};
    using type = fixed_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of -pi
   */
#if __cpp_nontype_template_args >= 201911L
  template<value::number T>
  using fixed_minus_pi = value::Fixed<T, -numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_minus_pi
  {
    static_assert(value::number<T>);
    using value_type = T;
    static constexpr value_type value {-numbers::pi_v<value_type>};
    using type = fixed_minus_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of 2*pi
   */
#if __cpp_nontype_template_args >= 201911L
  template<value::number T>
  using fixed_2pi = value::Fixed<T, 2 * numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_2pi
  {
    static_assert(value::number<T>);
    using value_type = T;
    static constexpr value_type value {2 * numbers::pi_v<value_type>};
    using type = fixed_2pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


/**
 * \brief A fixed version of pi/2
 */
#if __cpp_nontype_template_args >= 201911L
  template<value::number T>
  using fixed_half_pi = value::Fixed<T, static_cast<T>(0.5) * numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_half_pi
  {
    static_assert(value::number<T>);
    using value_type = T;
    static constexpr value_type value {static_cast<value_type>(0.5) * numbers::pi_v<value_type>};
    using type = fixed_half_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


/**
 * \brief A fixed version of -pi/2
 */
#if __cpp_nontype_template_args >= 201911L
  template<value::number T>
  using fixed_minus_half_pi = value::Fixed<T, static_cast<T>(-0.5) * numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_minus_half_pi
  {
    static_assert(value::number<T>);
    using value_type = T;
    static constexpr value_type value {static_cast<value_type>(-0.5) * numbers::pi_v<value_type>};
    using type = fixed_minus_half_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_CLASSES_FIXED_CONSTANTS_HPP
