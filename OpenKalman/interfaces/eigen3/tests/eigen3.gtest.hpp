/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

/**
 * \dir
 * \brief Tests for the Eigen3 interface.
 *
 * \file
 * \brief Header file for Eigen3 tests.
 */

#ifndef EIGEN3_GTEST_HPP
#define EIGEN3_GTEST_HPP

#include <gtest/gtest.h>
#include "basics/tests/tests.hpp"

#include "interfaces/eigen3/eigen3.hpp"


namespace OpenKalman
{
  using namespace OpenKalman::Eigen3;
}
using namespace OpenKalman;


namespace OpenKalman::test
{
  namespace detail
  {
    template<typename Arg>
#ifdef __cpp_concepts
    concept eigen_type = typed_matrix_nestable<Arg> or covariance_nestable<Arg>;
#else
    static constexpr bool eigen_type = typed_matrix_nestable<Arg> or covariance_nestable<Arg>;
#endif
  } // namespace detail


#ifdef __cpp_concepts
  template<detail::eigen_type Arg1, detail::eigen_type Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<detail::eigen_type<Arg1> and detail::eigen_type<Arg2>>>
#endif
    : ::testing::AssertionResult
  {
    static ::testing::AssertionResult
    compare(const Arg1& A, const Arg2& B, const Err& err)
    {
      if constexpr (std::is_arithmetic_v<Err>)
      if (A.isApprox(B, err) or (A.isMuchSmallerThan(1., err) and B.isMuchSmallerThan(1., err)))
      {
        return ::testing::AssertionSuccess();
      }

      if constexpr (detail::eigen_type<Err>)
      if (((A - B).cwiseAbs().array() - make_native_matrix(err).array()).maxCoeff() <= 0)
      {
        return ::testing::AssertionSuccess();
      }

      return ::testing::AssertionFailure() << std::endl << A << std::endl << "is not near" << std::endl <<
        B << std::endl;
    }


    TestComparison(const Arg1& A, const Arg2& B, const Err& err = 1e-6)
      : ::testing::AssertionResult {compare(A, B, err)} {};

   };


} // namespace OpenKalman::test

#endif //EIGEN3_GTEST_HPP
