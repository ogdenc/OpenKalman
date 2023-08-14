/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general functions relating to scalar types.
 */

#ifndef OPENKALMAN_SCALAR_FUNCTIONS_HPP
#define OPENKALMAN_SCALAR_FUNCTIONS_HPP

#include <complex>

namespace OpenKalman
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
    return interface::ScalarTraits<R>::make_complex(std::forward<decltype(re)>(re), std::forward<decltype(im)>(im));
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
    return interface::ScalarTraits<std::decay_t<T>>::make_complex(std::forward<decltype(re)>(re), std::forward<decltype(im)>(im));
  }


  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam epsilon_factor A factor to be multiplied by the epsilon
   * \return true if within the rounding tolerance, otherwise false
   */
  template<unsigned int epsilon_factor = 2, typename Arg1, typename Arg2>
  constexpr bool are_within_tolerance(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (complex_number<Arg1> or complex_number<Arg2>)
    {
      using std::real, std::imag;
      return are_within_tolerance<epsilon_factor>(real(arg1), real(arg2)) and
        are_within_tolerance<epsilon_factor>(imag(arg1), imag(arg2));
    }
    else if (arg1 != arg1 or arg2 != arg2) return false; // in case either argument is NaN
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      constexpr auto ep = epsilon_factor * std::numeric_limits<Diff>::epsilon();
      return -static_cast<Diff>(ep) <= diff and diff <= static_cast<Diff>(ep);
    }
  }


  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam Err The error
   * \return true if within the error, otherwise false
   */
  template<typename Arg1, typename Arg2, typename Err>
  constexpr bool are_within_tolerance(const Arg1& arg1, const Arg2& arg2, const Err& err)
  {
    if constexpr (complex_number<Arg1> or complex_number<Arg2>)
    {
      using std::real, std::imag;
      return are_within_tolerance(real(arg1), real(arg2), err) and are_within_tolerance(imag(arg1), imag(arg2), err);
    }
    else if (arg1 != arg1 or arg2 != arg2) return false; // in case either argument is NaN
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      return -static_cast<Diff>(err) <= diff and diff <= static_cast<Diff>(err);
    }

  }


} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_FUNCTIONS_HPP
