/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Basic utilities for OpenKalman testing.
 */

#ifndef OPENKALMAN_TESTS_HPP
#define OPENKALMAN_TESTS_HPP

#include <type_traits>
#include <array>
#include <tuple>

#include <gtest/gtest.h>

#include "basics/basics.hpp"


namespace OpenKalman::test
{
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err>
#else
  template<typename Arg1, typename Arg2, typename Err, typename = void>
#endif
  struct TestComparison;


  template<typename Arg1, typename Arg2, typename Err = double>
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    return TestComparison<Arg1, Arg2, Err> {arg1, arg2, err};
  }


  namespace detail
  {
    template<typename T>
    struct is_std_array : std::false_type {};

    template<typename T, std::size_t N>
    struct is_std_array<std::array<T, N>> : std::true_type {};

    template<typename T>
    static constexpr bool is_std_array_v = is_std_array<std::decay_t<T>>::value;
  } // namespace detail


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err> requires
    detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
    detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err>>>
#endif
    : ::testing::AssertionResult
  {

  private:

    template<typename A1, typename A2, std::size_t N>
    static ::testing::AssertionResult
    compare(const std::array<A1, N>& A, const std::array<A2, N>& B, const Err& e, const std::size_t i)
    {
      auto res_i = is_near(A[i], B[i], e);
      if (res_i)
      {
        if (i < N - 1) return compare(A, B, e, i + 1);
        else return ::testing::AssertionSuccess();
      }
      else
      {
        if (i < N - 1)
        {
          auto res_i1 = compare(A, B, e, i + 1);
          if (res_i1)
            return ::testing::AssertionFailure() << "array element " << i + 1 << "/" << N << ": " << res_i.message();
          else
            return ::testing::AssertionFailure() << "array element " << i + 1 << "/" << N << ": " << res_i.message() <<
              res_i1.message();
        }
        else
        {
          return ::testing::AssertionFailure() << "array element " << i + 1 << "/" << N << ": " << res_i.message();
        }
      }
    };

  public:

    TestComparison(const Arg1& A, const Arg2& B, const Err& e)
      : ::testing::AssertionResult {compare(A, B, e, 0)} {}
  };


  namespace detail
  {
    template<typename T>
    struct is_std_tuple : std::false_type {};

    template<typename...T>
    struct is_std_tuple<std::tuple<T...>> : std::true_type {};

    template<typename T>
    static constexpr bool is_std_tuple_v = is_std_tuple<std::decay_t<T>>::value;

  } // namespace detail


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err> requires
  detail::is_std_tuple_v<Arg1> and detail::is_std_tuple_v<Arg2> and std::is_arithmetic_v<Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
    template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
    detail::is_std_tuple_v<Arg1> and detail::is_std_tuple_v<Arg2> and std::is_arithmetic_v<Err>>>
#endif
    : ::testing::AssertionResult
  {

  private:

    static ::testing::AssertionResult compare(const std::tuple<>&, const std::tuple<>&, const Err& e)
    {
      return ::testing::AssertionSuccess();
    }


    template<typename Head1, typename Head2, typename ... Tail1, typename ... Tail2>
    static ::testing::AssertionResult compare(
      const std::tuple<Head1, Tail1...>& A, const std::tuple<Head2, Tail2...>& B, const Err& e)
    {
      static_assert(sizeof...(Tail1) == sizeof...(Tail2));

      auto res_head = is_near(std::get<0>(A), std::get<0>(B), e);

      if constexpr (sizeof...(Tail1) == 0)
      {
        return res_head;
      }
      else
      {
        auto res_tail = compare(
          OpenKalman::internal::tuple_slice<1, sizeof...(Tail1)>(A),
          OpenKalman::internal::tuple_slice<1, sizeof...(Tail2)>(B), e);

        if (res_head) return res_tail;
        else return ::testing::AssertionFailure() << res_head.message() << res_tail.message();
      }
    }

  public:

    TestComparison(const Arg1& A, const Arg2& B, const Err& e)
      : ::testing::AssertionResult {compare(A, B, e)} {}
  };


} // namespace OpenKalman::test


#endif //OPENKALMAN_TESTS_HPP
