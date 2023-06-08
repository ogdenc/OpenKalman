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

#include "basics/tests/tests.hpp"
#include "special-matrices/special-matrices.hpp"
#include "interfaces/eigen3/eigen3.hpp"


namespace OpenKalman::test
{
  using namespace OpenKalman;
  using namespace OpenKalman::Eigen3;


#ifdef __cpp_concepts
  template<native_eigen_dense Arg1, native_eigen_dense Arg2, typename Err> requires std::is_arithmetic_v<Err> or native_eigen_dense<Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<native_eigen_dense<Arg1> and native_eigen_dense<Arg2> and
    (std::is_arithmetic_v<Err> or native_eigen_dense<Err>)>>
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
    (native_eigen_general<Arg1> and not native_eigen_general<Arg2>) or
    (not native_eigen_general<Arg1> and native_eigen_general<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<indexible<Arg1> and indexible<Arg2> and
    ((native_eigen_general<Arg1> and not native_eigen_general<Arg2>) or
    (not native_eigen_general<Arg1> and native_eigen_general<Arg2>))>>
#endif
    : ::testing::AssertionResult
  {
  private:

    using A = std::conditional_t<native_eigen_general<Arg1>, Arg1, Arg2>;

  public:
    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {is_near(to_native_matrix<A>(arg1), to_native_matrix<A>(arg2), err)} {};

  };


} // namespace OpenKalman::test

#endif //EIGEN3_GTEST_HPP
