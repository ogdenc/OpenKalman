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
 * \brief Basic utilities for OpenKalman matrix testing.
 */

#ifndef OPENKALMAN_MATRIX_GTEST_HPP
#define OPENKALMAN_MATRIX_GTEST_HPP

#include "interfaces/eigen3/tests/eigen3.gtest.hpp"

#include "matrices/matrices.hpp"


namespace OpenKalman::test
{
  using namespace OpenKalman;


  namespace detail
  {
    template<typename Arg>
#ifdef __cpp_concepts
    concept nestable_type = typed_matrix_nestable<Arg> or covariance_nestable<Arg>;
#else
    static constexpr bool nestable_type = typed_matrix_nestable<Arg> or covariance_nestable<Arg>;
#endif


    template<typename Arg>
#ifdef __cpp_concepts
    concept matrix_type = typed_matrix<Arg> or covariance<Arg>;
#else
    static constexpr bool matrix_type = typed_matrix<Arg> or covariance<Arg>;
#endif
  } // namespace detail



#ifdef __cpp_concepts
  template<detail::matrix_type Arg1, detail::nestable_type Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<detail::matrix_type<Arg1> and detail::nestable_type<Arg2>>>
#endif
    : ::testing::AssertionResult
  {
    TestComparison(const Arg1& A, const Arg2& B, const Err& err)
      : ::testing::AssertionResult {is_near(make_native_matrix(A), B, err)} {};
  };


#ifdef __cpp_concepts
  template<detail::nestable_type Arg1, detail::matrix_type Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
    template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<detail::nestable_type<Arg1> and detail::matrix_type<Arg2>>>
#endif
    : ::testing::AssertionResult
  {
    TestComparison(const Arg1& A, const Arg2& B, const Err& err)
      : ::testing::AssertionResult {is_near(A, make_native_matrix(B), err)} {};
  };


#ifdef __cpp_concepts
  template<detail::matrix_type Arg1, detail::matrix_type Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
    template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<detail::matrix_type<Arg1> and detail::matrix_type<Arg2>>>
#endif
    : ::testing::AssertionResult
  {
    TestComparison(const Arg1& A, const Arg2& B, const Err& err)
      : ::testing::AssertionResult {is_near(make_native_matrix(A), make_native_matrix(B), err)} {};
  };


} // namespace OpenKalman::test


#endif //OPENKALMAN_MATRIX_GTEST_HPP
