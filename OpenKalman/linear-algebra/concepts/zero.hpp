/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref zero.
 */

#ifndef OPENKALMAN_ZERO_HPP
#define OPENKALMAN_ZERO_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_zero : std::false_type {};

    template<typename T>
    struct is_zero<T, std::enable_if_t<values::fixed<constant_coefficient<T>>>>
      : std::bool_constant<values::internal::near(constant_coefficient_v<T>, 0)> {};
  }
#endif


  /**
   * \brief Specifies that a type is known at compile time to be a constant matrix of value zero.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero =
    values::fixed<constant_coefficient<T>> and values::internal::near(constant_coefficient_v<T>, 0);
#else
  constexpr bool zero = detail::is_zero<T>::value;
#endif


}

#endif
