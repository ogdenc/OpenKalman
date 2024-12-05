/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref value::internal::make_complex_number function.
 */

#ifndef OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP
#define OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP

#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/complex.hpp"
#include "linear-algebra/values/functions/real.hpp"
#include "linear-algebra/values/functions/imag.hpp"

namespace OpenKalman::value::internal
{
  /**
   * \internal
   * \brief Make a complex number of type T from real and imaginary parts.
   * \tparam T The complex type of the result
   * \param re The real part.
   * \param im The imaginary part.
   */
#ifdef __cpp_concepts
  template<value::complex T, std::convertible_to<real_type_of_t<T>> Re, std::convertible_to<real_type_of_t<T>> Im = Re> requires
    value::value<Re> and value::value<Im> and (not value::complex<Re>) and (not value::complex<Im>)
#else
  template<typename T, typename Re, typename Im = Re, std::enable_if_t<value::complex<T> and
    std::is_convertible_v<Re, real_type_of_t<T>> and std::is_convertible_v<Im, real_type_of_t<T>> and
    value::value<Re> and value::value<Im> and (not value::complex<Re>) and (not value::complex<Im>), int> = 0>
#endif
  constexpr std::decay_t<T>
  make_complex_number(Re&& re, Im&& im = 0)
  {
    return interface::number_traits<std::decay_t<T>>::make_complex(std::forward<Re>(re), std::forward<Im>(im));
  }


  /**
   * \internal
   * \overload
   * \brief Convert a complex number of one real type from a complex number of another real type.
   * \tparam T The complex type of the result, or alternatively, its underlying real type
   * \tparam Arg A complex number to be converted.
   */
#ifdef __cpp_concepts
  template<value::value T, value::complex Arg> requires std::constructible_from<real_type_of_t<T>, real_type_of_t<Arg>>
  constexpr value::complex decltype(auto)
#else
  template<typename T, typename Arg, std::enable_if_t<value::value<T> and value::complex<Arg> and
    std::is_constructible_v<real_type_of_t<T>, real_type_of_t<Arg>>, int> = 0>
  constexpr decltype(auto)
#endif
  make_complex_number(Arg&& arg)
  {
    using R = real_type_of_t<T>;
    if constexpr (std::is_same_v<R, std::decay_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return interface::number_traits<std::decay_t<T>>::make_complex(
        static_cast<R>(value::real(arg)), static_cast<R>(value::imag(arg)));
    }
  }


/**
   * \internal
   * \overload
   * \brief Make a complex number from real and imaginary parts, deriving the complex type from the arguments.
   * \param re The real part.
   * \param im The imaginary part.
   */
#ifdef __cpp_concepts
  template<value::number Re, value::number Im> requires
    (not value::complex<Re>) and (not value::complex<Im>) and std::common_with<Re, Im>
  constexpr value::complex auto
#else
  template<typename Re, typename Im, std::enable_if_t<value::number<Re> and value::number<Im> and
    (not value::complex<Re>) and (not value::complex<Im>), int> = 0>
  constexpr auto
#endif
  make_complex_number(const Re& re, const Im& im = 0)
  {
    return interface::number_traits<std::decay_t<std::common_type_t<Re, Im>>>::make_complex(re, im);
  }


} // namespace OpenKalman::value::internal

#endif //OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP
