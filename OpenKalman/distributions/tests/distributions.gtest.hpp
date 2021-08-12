/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Basic utilities for OpenKalman distribution testing.
 */

#ifndef OPENKALMAN_DISTRIBUTIONS_GTEST_HPP
#define OPENKALMAN_DISTRIBUTIONS_GTEST_HPP

#include "matrices/tests/matrix.gtest.hpp"

#include "distributions/distributions.hpp"


namespace OpenKalman::test
{
  using namespace OpenKalman;

#ifdef __cpp_concepts
  template<gaussian_distribution Arg1, gaussian_distribution Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err,
    std::enable_if_t<gaussian_distribution<Arg1> and gaussian_distribution<Arg2>>>
#endif
    : ::testing::AssertionResult
  {

  private:

    static ::testing::AssertionResult
    compare(const Arg1& A, const Arg2& B, const Err& err)
    {
      if (is_near(mean_of(A), mean_of(B), err) and is_near(covariance_of(A), covariance_of(B), err))
      {
        return ::testing::AssertionSuccess();
      }
      else
      {
        return ::testing::AssertionFailure() << std::endl << A << std::endl << "is not near" << std::endl <<
          B << std::endl;
      }
    }

  public:

    TestComparison(const Arg1& A, const Arg2& B, const Err& err)
      : ::testing::AssertionResult {compare(A, B, err)} {};

  };

} // namespace OpenKalman::test


#endif //OPENKALMAN_DISTRIBUTIONS_GTEST_HPP
