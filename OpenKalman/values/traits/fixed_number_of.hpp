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
 * \brief Definition for \ref value::fixed_number_of.
 */

#ifndef OPENKALMAN_VALUES_FIXED_NUMBER_OF_HPP
#define OPENKALMAN_VALUES_FIXED_NUMBER_OF_HPP

#include <type_traits>
#include "values/concepts/value.hpp"
#include "values/concepts/fixed.hpp"

namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct fixed_number_of_impl : std::false_type {};

    template<typename T>
    struct fixed_number_of_impl<T, std::void_t<decltype(T::value)>> : std::true_type {};
  }
#endif


  /**
   * \brief The fixed number associated with a \ref value::fixed.
   */
#ifdef __cpp_concepts
  template<value::value T>
#else
  template<typename T, typename = void>
#endif
  struct fixed_number_of {};


#ifdef __cpp_concepts
  template<value::fixed T>
  struct fixed_number_of<T>
#else
  template<typename T>
  struct fixed_number_of<T, std::enable_if_t<value::fixed<T>>>
#endif
  {
  private:

    static constexpr auto get_value()
    {
#ifdef __cpp_concepts
      if constexpr (requires { std::decay_t<T>::value; })
#else
      if constexpr (detail::fixed_number_of_impl<std::decay_t<T>>::value)
#endif
        return std::decay_t<T>::value;
      else
        return std::decay_t<T>{}();
    };

  public:

    using value_type = std::decay_t<decltype(get_value())>;
    static constexpr value_type value {get_value()};
    using type = fixed_number_of;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };


  /**
   * \brief Helper template for \ref fixed_number_of.
   */
  template<typename T>
  constexpr auto fixed_number_of_v = fixed_number_of<T>::value;


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUES_FIXED_NUMBER_OF_HPP
