/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_H
#define OPENKALMAN_TESTS_H

#include <tuple>
#include <iostream>
#include <gtest/gtest.h>
#include "variables/interfaces/Eigen3.h"
#include "OpenKalman.h"

using namespace OpenKalman;

template<typename Arg>
static constexpr bool is_test_trait =
  is_typed_matrix_base_v<Arg> or
  is_typed_matrix_v<Arg> or
  is_covariance_base_v<Arg> or
  is_covariance_v<Arg>;

template<typename ArgA, typename ArgB, std::enable_if_t<is_test_trait<ArgA> and is_test_trait<ArgB>, int> = 0>
::testing::AssertionResult is_near(const ArgA& A, const ArgB& B, const double err = 1e-6)
{
  auto A_n = strict_matrix(A);
  auto B_n = strict_matrix(B);

  if (A_n.isApprox(B_n, err))
  {
    return ::testing::AssertionSuccess();
  }
  else
  {
    return ::testing::AssertionFailure() << std::endl << A_n << std::endl << "is not near" << std::endl << B_n << std::endl;
  }
}


template<typename ArgA, typename ArgB, typename Err,
  std::enable_if_t<is_test_trait<ArgA> and is_test_trait<ArgB> and is_Eigen_matrix_v<Err>, int> = 0>
::testing::AssertionResult is_near(const ArgA& A, const ArgB& B, const Err& err)
{
  auto A_n = strict_matrix(A);
  auto B_n = strict_matrix(B);

  if (((A_n - B_n).cwiseAbs().array() - strict_matrix(err).array()).maxCoeff() <= 0)
  {
    return ::testing::AssertionSuccess();
  }
  else
  {
    return ::testing::AssertionFailure() << std::endl << A_n << std::endl << "is not near" << std::endl << B_n << std::endl;
  }
}


template<
    typename Dist1,
    typename Dist2,
    std::enable_if_t<is_Gaussian_distribution_v<Dist1>, int> = 0,
    std::enable_if_t<is_Gaussian_distribution_v<Dist2>, int> = 0>
::testing::AssertionResult is_near(
    const Dist1& A,
    const Dist2& B,
    const typename DistributionTraits<Dist1>::Scalar err = (typename DistributionTraits<Dist1>::Scalar) 1e-6)
{
  if (is_near(mean(A), mean(B), err) and is_near(covariance(A), covariance(B), err))
  {
    return ::testing::AssertionSuccess();
  }
  else
  {
    return ::testing::AssertionFailure() << std::endl << A << std::endl << "is not near" << std::endl << B << std::endl;
  }
}


template<typename Arg1, typename Arg2, typename ... Rest1, typename ... Rest2>
::testing::AssertionResult is_near(
    const std::tuple<Arg1, Rest1...>& A,
    const std::tuple<Arg2, Rest2...>& B,
    const double e = 1e-6)
{
  auto a = std::get<0>(A);
  auto b = std::get<0>(B);
  if constexpr (sizeof...(Rest1) == 0)
  {
    return is_near(a, b, e);
  }
  else
  {
    const auto a_tail =
      std::apply([](auto&&, auto&& ...rest) { return std::tuple {std::forward<decltype(rest)>(rest)...}; }, A);
    const auto b_tail =
      std::apply([](auto&&, auto&& ...rest) { return std::tuple {std::forward<decltype(rest)>(rest)...}; }, B);
    if (is_near(a, b, e))
    {
      return is_near(a_tail, b_tail, e);
    }
    else
    {
      return ::testing::AssertionFailure() << is_near(a, b, e).message() << is_near(a_tail, b_tail, e).message();
    }
  }
}

inline ::testing::AssertionResult is_near(const std::tuple<>&, const std::tuple<>&, const double e = 1e-6)
{
  return ::testing::AssertionSuccess();
}



#endif //OPENKALMAN_TESTS_H
