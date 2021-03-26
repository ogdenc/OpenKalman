/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_HPP
#define OPENKALMAN_TESTS_HPP

#include <tuple>
#include <iostream>
#include <gtest/gtest.h>
#include "OpenKalman-Eigen3.hpp"

using namespace OpenKalman;

using std::numbers::pi;
using std::numbers::log2e;
using std::numbers::sqrt2;

#ifdef __cpp_concepts
template<typename Arg>
concept test_trait = typed_matrix_nestable<Arg> or typed_matrix<Arg> or covariance_nestable<Arg> or covariance<Arg>;
#else
template<typename Arg>
static constexpr bool test_trait =
  typed_matrix_nestable<Arg> or typed_matrix<Arg> or covariance_nestable<Arg> or covariance<Arg>;
#endif


#ifdef __cpp_concepts
template<test_trait ArgA, test_trait ArgB>
#else
template<typename ArgA, typename ArgB, std::enable_if_t<test_trait<ArgA> and test_trait<ArgB>, int> = 0>
#endif
::testing::AssertionResult is_near(ArgA&& A, ArgB&& B, const double err = 1e-6)
{
  auto A_n = make_native_matrix(std::forward<ArgA>(A));
  auto B_n = make_native_matrix(std::forward<ArgB>(B));

  if (A_n.isApprox(B_n, err) or (A_n.isMuchSmallerThan(1., err) and B_n.isMuchSmallerThan(1., err)))
  {
    return ::testing::AssertionSuccess();
  }
  else
  {
    return ::testing::AssertionFailure() << std::endl << A_n << std::endl << "is not near" << std::endl << B_n << std::endl;
  }
}


#ifdef __cpp_concepts
template<test_trait ArgA, test_trait ArgB, Eigen3::eigen_matrix Err>
#else
template<typename ArgA, typename ArgB, typename Err,
  std::enable_if_t<test_trait<ArgA> and test_trait<ArgB> and Eigen3::eigen_matrix<Err>, int> = 0>
#endif
::testing::AssertionResult is_near(ArgA&& A, ArgB&& B, const Err& err)
{
  auto A_n = make_native_matrix(std::forward<ArgA>(A));
  auto B_n = make_native_matrix(std::forward<ArgB>(B));

  if (((A_n - B_n).cwiseAbs().array() - make_native_matrix(err).array()).maxCoeff() <= 0)
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
    std::enable_if_t<gaussian_distribution<Dist1>, int> = 0,
    std::enable_if_t<gaussian_distribution<Dist2>, int> = 0>
::testing::AssertionResult is_near(
    Dist1&& A,
    Dist2&& B,
    const typename DistributionTraits<Dist1>::Scalar err = (typename DistributionTraits<Dist1>::Scalar) 1e-6)
{
  if (is_near(mean_of(A), mean_of(B), err) and is_near(covariance_of(A), covariance_of(B), err))
  {
    return ::testing::AssertionSuccess();
  }
  else
  {
    return ::testing::AssertionFailure() << std::endl << std::forward<Dist1>(A) <<
      std::endl << "is not near" << std::endl << std::forward<Dist2>(B) << std::endl;
  }
}

inline namespace
{
  template<typename Arg1, typename Arg2, std::size_t N>
  ::testing::AssertionResult
  is_near_impl(const std::array<Arg1, N>& A, const std::array<Arg2, N>& B, const double e, const std::size_t i)
  {
    auto res_i = is_near(A[i], B[i], e);
    if (res_i)
    {
      if (i < N - 1) return is_near_impl(A, B, e, i + 1);
      else return ::testing::AssertionSuccess();
    }
    else
    {
      if (i < N - 1)
      {
        auto res_i1 = is_near_impl(A, B, e, i + 1);
        if (res_i1) return ::testing::AssertionFailure() << "array element " << i+1 << "/" << N << ": " << res_i.message();
        else return ::testing::AssertionFailure() << "array element " << i+1 << "/" << N << ": " << res_i.message() << res_i1.message();
      }
      else
      {
        return ::testing::AssertionFailure() << "array element " << i+1 << "/" << N << ": " << res_i.message();
      }
    }
  };
}

template<typename Arg1, typename Arg2, std::size_t N>
::testing::AssertionResult is_near(
  const std::array<Arg1, N>& A,
  const std::array<Arg2, N>& B,
  const double e = 1e-6)
{
  return is_near_impl(A, B, e, 0);
}


inline ::testing::AssertionResult is_near(const std::tuple<>&, const std::tuple<>&, const double e = 1e-6)
{
  return ::testing::AssertionSuccess();
}

template<typename Arg1, typename Arg2, typename ... Rest1, typename ... Rest2>
::testing::AssertionResult is_near(
    const std::tuple<Arg1, Rest1...>& A,
    const std::tuple<Arg2, Rest2...>& B,
    const double e = 1e-6)
{
  static_assert(sizeof...(Rest1) == sizeof...(Rest2));
  auto a = std::get<0>(A);
  auto b = std::get<0>(B);
  if constexpr (sizeof...(Rest1) == 0)
  {
    return is_near(a, b, e);
  }
  else
  {
    const auto a_tail = internal::tuple_slice<1, sizeof...(Rest1)>(A);
    const auto b_tail = internal::tuple_slice<1, sizeof...(Rest2)>(B);
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


#endif //OPENKALMAN_TESTS_HPP
