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

#include "interfaces/eigen3/eigen3.hpp"

#include "basics/tests/tests.hpp"


namespace OpenKalman::test
{
  using namespace OpenKalman;
  using namespace OpenKalman::Eigen3;


#ifdef __cpp_concepts
  template<native_eigen_matrix Arg1, native_eigen_matrix Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<native_eigen_matrix<Arg1> and native_eigen_matrix<Arg2>>>
#endif
    : ::testing::AssertionResult
  {

  private:

    static ::testing::AssertionResult
    compare(const Arg1& arg1, const Arg2& arg2, const Err& err)
    {
      if constexpr (std::is_arithmetic_v<Err>)
        if (arg1.isApprox(arg2, err) or (arg1.isMuchSmallerThan(1., err) and arg2.isMuchSmallerThan(1., err)))
        {
          return ::testing::AssertionSuccess();
        }

      if constexpr (native_eigen_matrix<Err>)
        if (((arg1 - arg2).cwiseAbs().array() - make_native_matrix(err).array()).maxCoeff() <= 0)
        {
          return ::testing::AssertionSuccess();
        }

      return ::testing::AssertionFailure() << std::endl << arg1 << std::endl << "is not near" << std::endl <<
        arg2 << std::endl;
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err)} {};

   };


#ifdef __cpp_concepts
  template<eigen_native_general Arg1, eigen_native_general Arg2, typename Err>
  requires (not native_eigen_matrix<Arg1>) or (not native_eigen_matrix<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<eigen_native_general<Arg1> and eigen_native_general<Arg2> and
    (not native_eigen_matrix<Arg1>) or (not native_eigen_matrix<Arg2>)>>
#endif
    : ::testing::AssertionResult
  {
  private:
    template<typename Arg>
    decltype(auto) convert_impl(const Arg& arg)
    {
      if constexpr (native_eigen_matrix<Arg>)
        return arg;
      else
        return make_native_matrix(arg);
    }

  public:
    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {is_near(convert_impl(arg1), convert_impl(arg2), err)} {};

  };


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
  requires (not eigen_native_general<Arg1>) or (not eigen_native_general<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<detail::eigen_type<Arg1> and detail::eigen_type<Arg2> and
    (not eigen_native_general<Arg1>) or (not eigen_native_general<Arg2>)>>
#endif
    : ::testing::AssertionResult
  {
  private:
    template<typename Arg>
    decltype(auto) convert_impl(const Arg& arg)
    {
      if constexpr (native_eigen_matrix<Arg>)
        return arg;
      else
        return make_native_matrix(arg);
    }

  public:
    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {is_near(convert_impl(arg1), convert_impl(arg2), err)} {};

  };


} // namespace OpenKalman::test

#endif //EIGEN3_GTEST_HPP
