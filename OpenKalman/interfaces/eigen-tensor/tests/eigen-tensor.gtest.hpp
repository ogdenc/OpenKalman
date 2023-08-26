/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Tests for the Eigen Tensor module interface.
 *
 * \file
 * \brief Header file for Eigen Tensor module tests.
 */

#ifndef EIGEN_TENSOR_GTEST_HPP
#define EIGEN_TENSOR_GTEST_HPP

#include "interfaces/eigen-tensor/eigen-tensor.hpp"
#include "interfaces/eigen/tests/eigen.gtest.hpp"

namespace OpenKalman::test
{

/*#ifdef __cpp_concepts
  template<Eigen3::eigen_dense_general Arg1, Eigen3::eigen_dense_general Arg2, typename Err> requires
    std::is_arithmetic_v<Err> or Eigen3::eigen_dense_general<Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<Eigen3::eigen_dense_general<Arg1> and Eigen3::eigen_dense_general<Arg2> and
    (std::is_arithmetic_v<Err> or Eigen3::eigen_dense_general<Err>)>>
#endif
    : ::testing::AssertionResult
  {

  private:

    static ::testing::AssertionResult
    compare(const Arg1& arg1, const Arg2& arg2, const Err& err)
    {
      if constexpr (std::is_arithmetic_v<Err>)
      {
        if (arg1.matrix().isApprox(arg2.matrix(), err) or (arg1.matrix().isMuchSmallerThan(1., err) and
          arg2.matrix().isMuchSmallerThan(1., err)))
        {
          return ::testing::AssertionSuccess();
        }
      }
      else
      {
        if (((arg1.array() - arg2.array()).abs() - err).maxCoeff() <= 0)
        {
          return ::testing::AssertionSuccess();
        }
      }

      return ::testing::AssertionFailure() << std::endl << arg1 << std::endl << "is not near" << std::endl <<
        arg2 << std::endl;
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err)} {};

   };


#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2, typename Err> requires
    (Eigen3::eigen_general<Arg1, true> and not Eigen3::eigen_general<Arg2, true>) or
    (not Eigen3::eigen_general<Arg1, true> and Eigen3::eigen_general<Arg2, true>)
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<indexible<Arg1> and indexible<Arg2> and
    ((Eigen3::eigen_general<Arg1, true> and not Eigen3::eigen_general<Arg2, true>) or
    (not Eigen3::eigen_general<Arg1, true> and Eigen3::eigen_general<Arg2, true>))>>
#endif
    : ::testing::AssertionResult
  {
  private:

    using A = std::conditional_t<Eigen3::eigen_general<Arg1, true>, Arg1, Arg2>;

  public:
    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {is_near(to_native_matrix<A>(arg1), to_native_matrix<A>(arg2), err)} {};

  };*/


} // namespace OpenKalman::test

#endif //EIGEN_TENSOR_GTEST_HPP
