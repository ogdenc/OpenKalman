/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Basic utilities for OpenKalman testing.
 */

#ifndef OPENKALMAN_VALUES_TESTS_HPP
#define OPENKALMAN_VALUES_TESTS_HPP

#include <string>
#include "basics/tests/tests.hpp"
#include "values/concepts/value.hpp"
#include "values/functions/to_number.hpp"
#include "values/concepts/complex.hpp"
#include "values/functions/internal/near.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"

namespace OpenKalman::test
{
  /**
   * \internal
   * \brief Compare two values::value objects.
   */
#ifdef __cpp_concepts
  template<values::value Arg1, values::value Arg2, values::value Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<values::value<Arg1> and values::value<Arg2> and values::value<Err>>>
#endif
    : ::testing::AssertionResult
  {
  private:

    template<typename Arg>
    static auto print(Arg&& arg)
    {
      if constexpr (values::complex<Arg>)
      {
        return std::to_string(values::real(arg)) + " + " + std::to_string(values::imag(arg)) + "i";
      }
      else
      {
        return std::forward<Arg>(arg);
      }
    }


    static ::testing::AssertionResult
    compare(const Arg1 arg1, const Arg2 arg2, const Err& err)
    {
      if (values::internal::near(arg1, arg2, err))
        return ::testing::AssertionSuccess();
      else
        return ::testing::AssertionFailure() << print(values::to_number(arg2)) << " is not within " <<
          print(values::to_number(err)) << " of " << print(values::to_number(arg1));
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err)} {}
  };

}


#endif
