/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref values::fixed_value_of.
 */

#ifndef OPENKALMAN_VALUES_FIXED_VALUE_OF_HPP
#define OPENKALMAN_VALUES_FIXED_VALUE_OF_HPP

#include <type_traits>
#include "values/concepts/fixed.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct fixed_value_of_impl : std::false_type {};

    template<typename T>
    struct fixed_value_of_impl<T, std::void_t<decltype(T::value)>> : std::true_type {};
  }
#endif


  /**
   * \brief The fixed value associated with a \ref fixed.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct fixed_value_of {};


  /// \overload
#ifdef __cpp_concepts
  template<fixed T>
  struct fixed_value_of<T>
#else
  template<typename T>
  struct fixed_value_of<T, std::enable_if_t<fixed<T>>>
#endif
  {
  private:

    static constexpr auto get_value()
    {
#ifdef __cpp_concepts
      if constexpr (requires { std::decay_t<T>::value; })
#else
      if constexpr (detail::fixed_value_of_impl<std::decay_t<T>>::value)
#endif
        return std::decay_t<T>::value;
      else
        return std::decay_t<T>{}();
    };

  public:

    using value_type = std::decay_t<decltype(get_value())>;
    static constexpr value_type value {get_value()};
    using type = fixed_value_of;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };


  /**
   * \brief Helper template for \ref fixed_value_of.
   */
  template<typename T>
  constexpr auto fixed_value_of_v = fixed_value_of<T>::value;


}

#endif
