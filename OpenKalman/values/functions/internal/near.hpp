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
 * \brief Definition for \ref values::internal::near function.
 */

#ifndef OPENKALMAN_VALUES_NEAR_HPP
#define OPENKALMAN_VALUES_NEAR_HPP

#include <limits>
#include "basics/basics.hpp"
#include "values/concepts/complex.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/concepts/value.hpp"
#include "values/functions/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"

namespace OpenKalman::values::internal
{
  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam epsilon_factor A factor to be multiplied by the epsilon
   * \return true if within the rounding tolerance, otherwise false
   */
#ifdef __cpp_concepts
  template<unsigned int epsilon_factor = 2, value Arg1, value Arg2>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<unsigned int epsilon_factor = 2, typename Arg1, typename Arg2,
    std::enable_if_t<value<Arg1> and value<Arg2>, int> = 0>
  constexpr auto
#endif
  near(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (fixed<Arg1> or fixed<Arg2>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg1>& a1, const value_type_of_t<Arg2>& a2) const
        { return near<epsilon_factor>(a1, a2); } };
      return values::operation(Op{}, arg1, arg2);
    }
    else if constexpr (complex<Arg1> or complex<Arg2>)
    {
      return operation(
        std::logical_and{},
        near<epsilon_factor>(real(arg1), real(arg2)),
        near<epsilon_factor>(imag(arg1), imag(arg2)));
    }
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      auto ep = static_cast<Diff>(epsilon_factor * std::numeric_limits<Diff>::epsilon());
      return -ep <= diff and diff <= ep;
    }
  }


  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam Err The error
   * \return true if within the error, otherwise false
   */
#ifdef __cpp_concepts
  template<value Arg1, value Arg2, value Err>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<typename Arg1, typename Arg2, typename Err,
    std::enable_if_t<value<Arg1> and value<Arg2> and value<Err>, int> = 0>
  constexpr auto
#endif
  near(const Arg1& arg1, const Arg2& arg2, const Err& err)
  {
    if constexpr (fixed<Arg1> or fixed<Arg2>)
    {
      struct Op { constexpr auto operator()(
        const value_type_of_t<Arg1>& a1,
        const value_type_of_t<Arg2>& a2,
        const value_type_of_t<Err>& e) const
      { return near(a1, a2, e); } };
      return values::operation(Op{}, arg1, arg2, err);
    }
    else if constexpr (complex<Arg1> or complex<Arg2> or complex<Err>)
    {
      auto dr = real(arg2) - real(arg1);
      auto di = imag(arg2) - imag(arg1);
      auto er = real(err);
      auto ei = imag(err);
      return dr * dr + di * di <= er * er + ei * ei;
    }
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      auto ep = static_cast<Diff>(err);
      return -ep <= diff and diff <= ep;
    }

  }


}

#endif
