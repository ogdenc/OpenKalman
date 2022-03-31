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
#include <iostream>

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


  // ---------- //
  //  Matrices  //
  // ---------- //

#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2, typename Err = double> requires
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, typename Err = double, std::enable_if_t<
    indexible<Arg1> and indexible<Arg2> and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value),
    int> = 0>
#endif
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    if constexpr (has_dynamic_dimensions<Arg1> or has_dynamic_dimensions<Arg2>)
      if (runtime_dimension_of<0>(arg1) != runtime_dimension_of<0>(arg2) or runtime_dimension_of<1>(arg1) != runtime_dimension_of<1>(arg2))
    {
      return ::testing::AssertionFailure() << std::endl << make_dense_writable_matrix_from(arg1) << std::endl <<
        "(rows " << runtime_dimension_of<0>(arg1) << ", cols " << runtime_dimension_of<1>(arg1) << "), is not near" << std::endl <<
        make_dense_writable_matrix_from(arg2) << std::endl << "(rows " << runtime_dimension_of<0>(arg2) << ", cols " << runtime_dimension_of<1>(arg2) <<
        ")" << std::endl;
    }

    return TestComparison<Arg1, Arg2, Err> {arg1, arg2, err};
  }


  // -------- //
  //  Arrays  //
  // -------- //

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
  template<typename Arg1, typename Arg2, typename Err = double>
  requires detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err> and
    (std::tuple_size_v<Arg1> == std::tuple_size_v<Arg2>) and
    (dynamic_rows<typename Arg1::value_type> or dynamic_rows<typename Arg2::value_type> or
      row_dimension_of_v<typename Arg1::value_type> == row_dimension_of_v<typename Arg2::value_type>) and
    (dynamic_columns<typename Arg1::value_type> or dynamic_columns<typename Arg2::value_type> or
      column_dimension_of_v<typename Arg1::value_type> == column_dimension_of_v<typename Arg2::value_type>)
#else
  template<typename Arg1, typename Arg2, typename Err = double, std::enable_if_t<
    detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err> and
    (std::tuple_size<Arg1>::value == std::tuple_size<Arg2>::value) and
    (dynamic_rows<typename Arg1::value_type> or dynamic_rows<typename Arg2::value_type> or
      row_dimension_of<typename Arg1::value_type>::value == row_dimension_of<typename Arg2::value_type>::value) and
    (dynamic_columns<typename Arg1::value_type> or dynamic_columns<typename Arg2::value_type> or
      column_dimension_of<typename Arg1::value_type>::value == column_dimension_of<typename Arg2::value_type>::value),
      int> = 0>
#endif
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    if constexpr (std::tuple_size_v<Arg1> != std::tuple_size_v<Arg2>)
    {
      return ::testing::AssertionFailure() << std::endl << "Size of first array (" <<
        std::tuple_size_v<Arg1> << " elements) does not match size of second array (" <<
        std::tuple_size_v<Arg2> << " elements)" << std::endl;
    }
    else
    {
      return TestComparison<Arg1, Arg2, Err> {arg1, arg2, err};
    }

    if constexpr (dynamic_rows<std::tuple_element_t<0, Arg1>> or dynamic_rows<std::tuple_element_t<0, Arg2>>)
      if (runtime_dimension_of<0>(std::get<0>(arg1)) != runtime_dimension_of<0>(std::get<0>(arg2)))
        throw std::logic_error {"Row dimension mismatch: " + std::to_string(runtime_dimension_of<0>(std::get<0>(arg1))) + " != " +
          std::to_string(runtime_dimension_of<0>(std::get<0>(arg2))) + " in is_near(array) of " + std::string {__FILE__}};

    if constexpr (dynamic_columns<std::tuple_element_t<0, Arg1>> or dynamic_columns<std::tuple_element_t<0, Arg2>>)
      if (runtime_dimension_of<1>(std::get<0>(arg1)) != runtime_dimension_of<1>(std::get<0>(arg2)))
        throw std::logic_error {"Column dimension mismatch: " + std::to_string(runtime_dimension_of<1>(std::get<0>(arg1))) + " != " +
          std::to_string(runtime_dimension_of<1>(std::get<0>(arg2))) + " in is_near(array) of " + std::string {__FILE__}};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err>
  requires detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err>
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


  // -------- //
  //  Tuples  //
  // -------- //

  namespace detail
  {
    template<typename T>
    struct is_std_tuple : std::false_type {};

    template<typename...T>
    struct is_std_tuple<std::tuple<T...>> : std::true_type {};

    template<typename T>
    static constexpr bool is_std_tuple_v = is_std_tuple<std::decay_t<T>>::value;

    template<typename T1, typename T2>
    struct tuple_sizes_match;

    template<>
    struct tuple_sizes_match<std::tuple<>, std::tuple<>> { static constexpr bool value = true; };

    template<typename Arg1, typename...Args1, typename Arg2, typename...Args2>
    struct tuple_sizes_match<std::tuple<Arg1, Args1...>, std::tuple<Arg2, Args2...>>
    {
      static constexpr bool value =
        (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
        (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>) and
        tuple_sizes_match<std::tuple<Args1...>, std::tuple<Args2...>>::value;
    };

  } // namespace detail


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err = double>
  requires detail::is_std_tuple_v<Arg1> and detail::is_std_tuple_v<Arg2> and std::is_arithmetic_v<Err> and
    detail::tuple_sizes_match<Arg1, Arg2>::value
#else
  template<typename Arg1, typename Arg2, typename Err = double, std::enable_if_t<
    detail::is_std_tuple_v<Arg1> and detail::is_std_tuple_v<Arg2> and std::is_arithmetic_v<Err> and
    detail::tuple_sizes_match<Arg1, Arg2>::value, int> = 0>
#endif
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    return TestComparison<Arg1, Arg2, Err> {arg1, arg2, err};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err>
  requires detail::is_std_tuple_v<Arg1> and detail::is_std_tuple_v<Arg2> and std::is_arithmetic_v<Err>
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
