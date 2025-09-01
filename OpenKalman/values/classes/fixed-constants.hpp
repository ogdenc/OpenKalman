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
 * \brief Definition of \ref values::fixed constants.
 */

#ifndef OPENKALMAN_VALUES_CLASSES_FIXED_CONSTANTS_HPP
#define OPENKALMAN_VALUES_CLASSES_FIXED_CONSTANTS_HPP

#include "values/concepts/number.hpp"
#include "values/classes/fixed_value.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A fixed version of pi
   */
#if __cpp_nontype_template_args >= 201911L
  template<number T>
  using fixed_pi = fixed_value<T, stdcompat::numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_pi
  {
    static_assert(number<T>);
    using value_type = T;
    static constexpr value_type value {stdcompat::numbers::pi_v<value_type>};
    using type = fixed_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of -pi
   */
#if __cpp_nontype_template_args >= 201911L
  template<number T>
  using fixed_minus_pi = fixed_value<T, -stdcompat::numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_minus_pi
  {
    static_assert(number<T>);
    using value_type = T;
    static constexpr value_type value {-stdcompat::numbers::pi_v<value_type>};
    using type = fixed_minus_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of 2*pi
   */
#if __cpp_nontype_template_args >= 201911L
  template<number T>
  using fixed_2pi = fixed_value<T, 2 * stdcompat::numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_2pi
  {
    static_assert(number<T>);
    using value_type = T;
    static constexpr value_type value {2 * stdcompat::numbers::pi_v<value_type>};
    using type = fixed_2pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of pi/2
   */
#if __cpp_nontype_template_args >= 201911L
  template<number T>
  using fixed_half_pi = fixed_value<T, static_cast<T>(0.5) * stdcompat::numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_half_pi
  {
    static_assert(number<T>);
    using value_type = T;
    static constexpr value_type value {static_cast<value_type>(0.5) * stdcompat::numbers::pi_v<value_type>};
    using type = fixed_half_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of -pi/2
   */
#if __cpp_nontype_template_args >= 201911L
  template<number T>
  using fixed_minus_half_pi = fixed_value<T, static_cast<T>(-0.5) * stdcompat::numbers::pi_v<T>>;
#else
  template<typename T>
  struct fixed_minus_half_pi
  {
    static_assert(number<T>);
    using value_type = T;
    static constexpr value_type value {static_cast<value_type>(-0.5) * stdcompat::numbers::pi_v<value_type>};
    using type = fixed_minus_half_pi;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };
#endif


  /**
   * \brief A fixed version of std::partial_ordering::equivalent
   */
  struct fixed_partial_ordering_equivalent
  {
    using value_type = stdcompat::partial_ordering;
    static constexpr auto value = value_type::equivalent;
    using type = fixed_partial_ordering_equivalent;
    constexpr operator value_type () const { return value; }
    constexpr value_type operator()() const { return value; }
  };


  /**
   * \brief A fixed version of std::partial_ordering::less
   */
  struct fixed_partial_ordering_less
  {
    using value_type = stdcompat::partial_ordering;
    static constexpr auto value = value_type::less;
    using type = fixed_partial_ordering_equivalent;
    constexpr operator value_type () const { return value; }
    constexpr value_type operator()() const { return value; }
  };


  /**
   * \brief A fixed version of std::partial_ordering::greater
   */
  struct fixed_partial_ordering_greater
  {
    using value_type = stdcompat::partial_ordering;
    static constexpr auto value = value_type::greater;
    using type = fixed_partial_ordering_equivalent;
    constexpr operator value_type () const { return value; }
    constexpr value_type operator()() const { return value; }
  };


  /**
   * \brief A fixed version of std::partial_ordering::unordered
   */
  struct fixed_partial_ordering_unordered
  {
    using value_type = stdcompat::partial_ordering;
    static constexpr auto value = value_type::unordered;
    using type = fixed_partial_ordering_equivalent;
    constexpr operator value_type () const { return value; }
    constexpr value_type operator()() const { return value; }
  };

}


#endif
