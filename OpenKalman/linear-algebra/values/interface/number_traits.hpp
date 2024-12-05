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
 * \brief Traits for arithmetic and complex scalar types.
 */

#ifndef OPENKALMAN_NUMBER_TRAITS_HPP
#define OPENKALMAN_NUMBER_TRAITS_HPP

#include <limits>
#include <complex>
#include "basics/language-features.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief Traits for numerical types, including user-defined types.
   * \details This is effectively an extension of std::numeric_limits<T>.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct number_traits
  {
    /**
     * This value is true for all T for which there exists a specialization of numeric_traits.
     */
    static constexpr bool is_specialized = false;


    /**
     * \brief Whether T is a complex number.
     */
    static constexpr bool is_complex = false;


    /**
     * \brief The underlying real type.
     */
    using real_type = T;


    /**
     * \brief Make a complex number consistent with T from two real arguments.
     * \tparam Re Real part
     * \tparam Im Imaginary part
     */
#ifdef __cpp_concepts
    template<std::convertible_to<real_type> Re, std::convertible_to<real_type> Im>
#else
    template<typename Re, typename Im, std::enable_if_t<
      std::is_convertible_v<Re, real_type> and std::is_convertible_v<Im, real_type>, int> = 0>
#endif
    static constexpr auto make_complex(Re&& re, Im&& im);


    /**
     * \brief The real part, if complex.
     * \details This is optional, and need only be defined if there is not otherwise a constexpr definition of real(t).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const T&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const T&>, int> = 0>
#endif
    static constexpr auto real(Arg&& arg);


    /*
     * \brief The imaginary part, if complex.
     * \details This is optional, and need only be defined if there is not otherwise a constexpr definition of imag(t).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const T&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const T&>, int> = 0>
#endif
    static constexpr auto imag(Arg&& arg);

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

    using real_type = std::conditional_t<std::is_integral_v<T>, double, T>;

    static constexpr std::complex<real_type> make_complex(real_type re, real_type im) { return {re, im}; }

  };


  /**
   * \brief Traits for std::complex.
   */
  template<typename T>
  struct number_traits<std::complex<T>>
  {
    static constexpr bool is_specialized = true;

    static constexpr bool is_complex = true;

    using real_type = T;

    static constexpr std::complex<T> make_complex(real_type re, real_type im) { return {re, im}; }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_NUMBER_TRAITS_HPP
