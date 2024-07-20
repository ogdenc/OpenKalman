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
 * \brief Traits for arithmetic and complex scalar types.
 */

#ifndef OPENKALMAN_SCALAR_TRAITS_HPP
#define OPENKALMAN_SCALAR_TRAITS_HPP

#include <complex>

namespace OpenKalman::interface
{

  /**
   * \internal
   * \brief Traits for scalar types, including user-defined scalar types.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct scalar_traits
  {
    /**
     * \brief Whether T is a complex number.
     */
    static constexpr bool is_complex = false;


    /**
     * \brief Make a complex number consistent with T from two real arguments.
     * \tparam Re Real part
     * \tparam Im Imaginary part
     */
    template<typename Re, typename Im>
    static constexpr decltype(auto) make_complex(Re&& re, Im&& im) = delete;


    /**
     * \brief The real part, if complex.
     * \details This is optional, and need only be defined if there is not otherwise a constexpr definition of real(t).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::decay_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::decay_t<T>&>, int> = 0>
#endif
    static constexpr auto real(Arg&& arg) = delete;


    /*
     * \brief The imaginary part, if complex.
     * \details This is optional, and need only be defined if there is not otherwise a constexpr definition of imag(t).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::decay_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::decay_t<T>&>, int> = 0>
#endif
    static constexpr auto imag(Arg&& arg) = delete;

  };



  /**
   * \brief Traits for std::is_arithmetic types
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_arithmetic_v<T>
  struct scalar_traits<T>
#else
  template<typename T>
  struct scalar_traits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
  {
    static constexpr bool is_complex = false;

    template<typename Re, typename Im>
    static constexpr decltype(auto) make_complex(Re&& re, Im&& im)
    {
      return std::complex<std::decay_t<T>>{std::forward<Re>(re), std::forward<Im>(im)};
    }
  };


  /**
   * \brief Traits for std::complex types
   */
  template<typename T>
  struct scalar_traits<std::complex<T>>
  {
  private:

    using Scalar = std::conditional_t<std::is_integral_v<T>, double, T>;
    static constexpr auto pi = numbers::pi_v<Scalar>;

  public:

    static constexpr bool is_complex = true;

    template<typename Re, typename Im>
    static constexpr std::complex<T> make_complex(Re&& re, Im&& im)
    {
      return {static_cast<T>(std::forward<Re>(re)), static_cast<T>(std::forward<Im>(im))};
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_SCALAR_TRAITS_HPP
