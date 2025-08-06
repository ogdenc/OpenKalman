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
 * \brief Definition for \values::complex.
 */

#ifndef OPENKALMAN_VALUE_COMPLEX_HPP
#define OPENKALMAN_VALUE_COMPLEX_HPP

#include <type_traits>
#include "values/interface/number_traits.hpp"
#include "values/traits/number_type_of.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct complex_impl : std::false_type {};


    template<typename T>
    struct complex_impl<T, std::enable_if_t<interface::number_traits<values::number_type_of_t<T>>::is_complex>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a values::value that reduces to std::complex or a custom complex type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex = values::value<T> and interface::number_traits<values::number_type_of_t<T>>::is_complex;
#else
  constexpr bool complex = detail::complex_impl<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUE_COMPLEX_HPP
