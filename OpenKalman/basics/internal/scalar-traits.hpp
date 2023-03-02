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

  // ------------------ //
  //    ScalarTraits    //
  // ------------------ //

  /**
   * \internal
   * \brief Traits for scalar types, including user-defined scalar types.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ScalarTraits
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
     * \brief Return a tuple containing the coefficients of the scalar value when viewed as a real vector space.
     * \detail Must be a constexpr function. Examples:
     * - If the scalar is an integral or floating value, the tuple contains only the value.
     * - If the scalar is complex, the tuple contains the real and imaginary components.
     * \return
     */
    static constexpr auto parts(std::decay_t<T> t) = delete;


    /**
     * \brief The real part, if complex.
     * \details This is optional, and need only be defined if there is not otherwise a constexpr definition of real(t).
     */
    static constexpr auto real(std::decay_t<T> t) = delete;


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
   * \overload
   * \brief Traits for std::is_arithmetic types
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_arithmetic_v<T>
  struct ScalarTraits<T>
#else
  template<typename T>
  struct ScalarTraits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
  {
    static constexpr bool is_complex = false;

    template<typename Re, typename Im>
    static constexpr decltype(auto) make_complex(Re&& re, Im&& im)
    {
      return std::complex<std::decay_t<T>>{std::forward<Re>(re), std::forward<Im>(im)};
    }

    template<typename Arg>
    static constexpr auto parts(Arg&& arg) { return std::forward_as_tuple(std::forward<Arg>(arg)); }
  };


  /**
   * \overload
   * \brief Traits for std::complex types
   */
  template<typename T>
  struct ScalarTraits<std::complex<T>>
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

    static constexpr std::tuple<Scalar, Scalar> parts(const std::complex<T>& arg) { return {std::real(arg), std::imag(arg)}; }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_SCALAR_TRAITS_HPP
