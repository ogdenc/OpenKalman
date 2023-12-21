/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \complex_number.
 */

#ifndef OPENKALMAN_COMPLEX_NUMBER_HPP
#define OPENKALMAN_COMPLEX_NUMBER_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_complex_number : std::false_type {};


    template<typename T>
    struct is_complex_number<T, std::enable_if_t<interface::scalar_traits<std::decay_t<T>>::is_complex>> : std::true_type {};
  }
#endif


  /**
   * \brief T is std::complex or a custom complex type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex_number = interface::scalar_traits<std::decay_t<T>>::is_complex;
#else
  constexpr bool complex_number = detail::is_complex_number<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COMPLEX_NUMBER_HPP
