/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#ifndef EIGEN3_GTEST_HPP
#define EIGEN3_GTEST_HPP

#include <tuple>
#include <iostream>
#include <gtest/gtest.h>
#include "interfaces/eigen3/eigen3.hpp"

namespace OpenKalman
{
  using namespace OpenKalman::Eigen3;
}

using namespace OpenKalman;


template<typename Arg>
static constexpr bool is_test_trait = typed_matrix_base<Arg> or covariance_base<Arg>;

template<typename ArgA, typename ArgB, std::enable_if_t<is_test_trait<ArgA> and is_test_trait<ArgB>, int> = 0>
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


template<typename ArgA, typename ArgB, typename Err,
  std::enable_if_t<is_test_trait<ArgA> and is_test_trait<ArgB> and Eigen3::eigen_matrix<Err>, int> = 0>
inline ::testing::AssertionResult is_near(ArgA&& A, ArgB&& B, const Err& err)
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
    const auto a_tail = OpenKalman::internal::tuple_slice<1, sizeof...(Rest1)>(A);
    const auto b_tail = OpenKalman::internal::tuple_slice<1, sizeof...(Rest2)>(B);
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


struct eigen3 : public ::testing::Test
{
  eigen3() {}

  void SetUp() override {}

  void TearDown() override {}

  ~eigen3() override {}
};


#endif //EIGEN3_GTEST_HPP
