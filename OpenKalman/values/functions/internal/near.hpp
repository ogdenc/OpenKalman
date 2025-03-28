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
 * \brief Definition for \ref value::internal::near function.
 */

#ifndef OPENKALMAN_ARE_WITHIN_TOLERANCE_HPP
#define OPENKALMAN_ARE_WITHIN_TOLERANCE_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/complex.hpp"
#include "values/functions/to_number.hpp"


namespace OpenKalman::value::internal
{
  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam epsilon_factor A factor to be multiplied by the epsilon
   * \return true if within the rounding tolerance, otherwise false
   */
  template<unsigned int epsilon_factor = 2, typename Arg1, typename Arg2>
  constexpr bool near(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (not value::number<Arg1> or not value::number<Arg2>)
    {
      return near(value::to_number<epsilon_factor>(arg1), value::to_number(arg2));
    }
    else if constexpr (value::complex<Arg1> or value::complex<Arg2>)
    {
      using std::real, std::imag;
      return
        near<epsilon_factor>(real(arg1), real(arg2)) and
        near<epsilon_factor>(imag(arg1), imag(arg2));
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
  constexpr bool near(const Arg1& arg1, const Arg2& arg2, const Err& err)
  {
    if constexpr (not value::number<Arg1> or not value::number<Arg2> or not value::number<Err>)
    {
      return near(value::to_number(arg1), value::to_number(arg2), value::to_number(err));
    }
    else if constexpr (value::complex<Arg1> or value::complex<Arg2> or value::complex<Err>)
    {
      using std::real, std::imag;
      auto dr = real(arg2) - real(arg1);
      auto di = imag(arg2) - imag(arg1);
      auto er = real(err);
      auto ei = imag(err);
      return dr * dr + di * di <= er * er + ei * ei;
    }
    else if (arg1 != arg1 or arg2 != arg2) return false; // in case either argument is NaN
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      return -static_cast<Diff>(err) <= diff and diff <= static_cast<Diff>(err);
    }

  }


} // namespace OpenKalman::value::internal

#endif //OPENKALMAN_ARE_WITHIN_TOLERANCE_HPP
