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
 * \internal
 * \brief Definition for make_complex_number function.
 */

#ifndef OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP
#define OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP

#include <complex>

namespace OpenKalman::internal
{
  /**
   * \brief Make a complex number from real and imaginary parts.
   * \param re The real part.
   * \param im The imaginary part.
   */
#ifdef __cpp_concepts
  constexpr complex_number auto
  make_complex_number(scalar_type auto&& re, scalar_type auto&& im = 0)
    requires (not complex_number<decltype(re)>) and std::same_as<std::decay_t<decltype(re)>, std::decay_t<decltype(im)>>
#else
  template<typename Re, typename Im, std::enable_if_t<scalar_type<Re> and (not complex_number<Re>) and
    std::is_same_v<std::decay_t<Re>, std::decay_t<Im>>, int> = 0>
  constexpr auto make_complex_number(Re&& re, Im&& im)
#endif
  {
    using R = std::decay_t<decltype(re)>;
    return interface::scalar_traits<R>::make_complex(std::forward<decltype(re)>(re), std::forward<decltype(im)>(im));
  }


  /**
   * \brief Make a complex number of type T from real and imaginary parts.
   * \tparam T A complex or floating type
   * \param re The real part.
   * \param im The imaginary part.
   */
#ifdef __cpp_concepts
  template<typename T>
  constexpr complex_number auto
  make_complex_number(scalar_type auto&& re, scalar_type auto&& im = 0)
  requires std::same_as<std::decay_t<decltype(re)>, std::decay_t<decltype(im)>>
#else
  template<typename T, typename Re, typename Im, std::enable_if_t<scalar_type<Re> and
    std::is_same_v<std::decay_t<Re>, std::decay_t<Im>>, int> = 0>
  constexpr auto make_complex_number(Re&& re, Im&& im)
#endif
  {
    return interface::scalar_traits<std::decay_t<T>>::make_complex(std::forward<decltype(re)>(re), std::forward<decltype(im)>(im));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP
