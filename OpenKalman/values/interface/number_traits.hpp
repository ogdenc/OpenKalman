/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Traits for arithmetic and complex scalar types.
 */

#ifndef OPENKALMAN_NUMBER_TRAITS_HPP
#define OPENKALMAN_NUMBER_TRAITS_HPP

#include <limits>
#include <complex>
#include "basics/basics.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief Traits for numerical types, including user-defined types.
   * \details This is effectively an extension of std::numeric_limits<T>.
   * \tparam T A cv- and ref-guaranteed numeric type.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct number_traits
  {
    /**
     * \brief This value is true for all T for which there exists a specialization of numeric_traits.
     * \details T is considered to be a \ref number if and only if this is true.
     */
    static constexpr bool is_specialized = false;


    /**
     * \brief Whether T is a complex number.
     */
    static constexpr bool is_complex = false;


#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /**
     * \brief A callable object that returns the real part of the argument of type T.
     */
    static constexpr auto real = [](T t) { return std::real(std::move(t)); };


    /**
     * \brief A callable object that returns the real part of the argument of type T.
     */
    static constexpr auto imag = [](T t) { return std::imag(std::move(t)); };


    /**
     * \brief A callable object that makes a complex number consistent with T from two real arguments.
     * \param re Real part
     * \param im Imaginary part
     */
    static constexpr auto make_complex = [](auto re, auto im) { return std::complex<T> {std::move(re), std::move(im)}; };

#endif
  };


  /**
   * \brief Traits for std::is_arithmetic types.
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_arithmetic_v<T>
  struct number_traits<T>
#else
  template<typename T>
  struct number_traits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
  {
    static constexpr bool is_specialized = true;
    static constexpr bool is_complex = false;
    static constexpr auto real = [](T t) { return std::real(std::move(t)); };
    static constexpr auto imag = [](T t) { return std::imag(std::move(t)); };
    static constexpr auto make_complex = [](T re, T im) { return std::complex<T> {std::move(re), std::move(im)}; };
  };


  /**
   * \brief Traits for std::complex.
   */
  template<typename T>
  struct number_traits<std::complex<T>>
  {
    static constexpr bool is_specialized = true;
    static constexpr bool is_complex = true;
    static constexpr auto real = [](std::complex<T> t) { return std::real(std::move(t)); };
    static constexpr auto imag = [](std::complex<T> t) { return std::imag(std::move(t)); };
    static constexpr auto make_complex = [](T re, T im) { return std::complex<T> {std::move(re), std::move(im)}; };
  };

}

#endif
