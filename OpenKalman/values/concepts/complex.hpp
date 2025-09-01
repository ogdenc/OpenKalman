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
 * \brief Definition for \ref values::complex.
 */

#ifndef OPENKALMAN_VALUES_COMPLEX_HPP
#define OPENKALMAN_VALUES_COMPLEX_HPP

#include "basics/basics.hpp"
#include "values/interface/number_traits.hpp"
#include "values/traits/value_type_of.hpp"
#include "value.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct complex_impl : std::false_type {};


    template<typename T>
    struct complex_impl<T, std::enable_if_t<interface::number_traits<typename values::value_type_of<T>::type>::is_complex>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a \ref value that reduces to std::complex or a custom complex type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex = value<T> and interface::number_traits<typename value_type_of<T>::type>::is_complex;
#else
  constexpr bool complex = value<T> and detail::complex_impl<std::decay_t<T>>::value;
#endif


}

#endif
