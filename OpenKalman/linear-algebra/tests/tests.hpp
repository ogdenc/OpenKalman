/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Basic utilities for OpenKalman testing.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_TESTS_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_TESTS_HPP

#include "collections/tests/tests.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra//property-functions//get_pattern_collection.hpp"
#include "linear-algebra/functions/n_ary_operation.hpp"
#include "linear-algebra/functions/reduce.hpp"

namespace OpenKalman::test
{
#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2, typename Err> requires
    (std::is_arithmetic_v<Err> or indexible<Err>) and
    (not collections::collection<Arg1> or not collections::collection<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
    template<typename Arg1, typename Arg2, typename Err>
    struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
      indexible<Arg1> and indexible<Arg2> and
      (std::is_arithmetic_v<Err> or indexible<Err>) and
      (not collections::collection<Arg1> or not collections::collection<Arg2>)>>
#endif
    : ::testing::AssertionResult
  {
  private:

    template<typename Arg, std::size_t...Is>
    inline ::testing::AssertionResult
    print_dimensions(::testing::AssertionResult& res, const Arg& arg, std::index_sequence<Is...>)
    {
      return (res << ... << (std::to_string(get_index_dimension_of<Is>(arg)) + (Is + 1 < sizeof...(Is) ? "," : "")));
    }


    static ::testing::AssertionResult
    compare(const Arg1& arg1, const Arg2& arg2, const Err& err)
    {
      static_assert(vector_space_descriptors_may_match_with<Arg1, Arg2>, "Dimensions must match");

      if constexpr (has_dynamic_dimensions<Arg1> or has_dynamic_dimensions<Arg2>)
        if (not vector_space_descriptors_match(arg1, arg2))
        {
          auto ret = ::testing::AssertionFailure();
          ret << "Dimensions of first argument (";
          detail::print_dimensions(ret, arg1, std::make_index_sequence<index_count_v<Arg1>>{});
          ret << ") and second argument (";
          detail::print_dimensions(ret, arg2, std::make_index_sequence<index_count_v<Arg2>>{});
          return ret << ") do not match";
        }

      if constexpr (std::is_arithmetic_v<Err>)
      {
        auto lhs = reduce(std::plus{}, n_ary_operation([](const auto& a, const auto& b){ auto c = a - b; return c * c; }, arg1, arg2));
        auto sum1 = reduce(std::plus{}, n_ary_operation([](const auto& a){ return a * a; }, arg1));
        auto sum2 = reduce(std::plus{}, n_ary_operation([](const auto& a){ return a * a; }, arg2));
        auto err2 = err * err;
        auto rhs = err2 * std::min(sum1, sum2);
        if (lhs <= rhs or (sum1 <= err2 and sum2 <= err2)) return ::testing::AssertionSuccess();
      }
      else
      {
        auto err2 = reduce(
          [](const auto& a, const auto& b){ return std::max(a, b); },
          n_ary_operation([](const auto& a, const auto& b, const auto& e){ return values::abs(a - b) - e; }, arg1, arg2, err));
        if (err2 <= 0) return ::testing::AssertionSuccess();
      }

      return ::testing::AssertionFailure() << std::endl << arg1 << std::endl << "is not near" << std::endl << arg2 << std::endl;
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err)} {};

  };







  // ------------------- //
  //  indexible objects  //
  // ------------------- //

#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2, typename Err = double>
#else
  template<typename Arg1, typename Arg2, typename Err = double, std::enable_if_t<
    indexible<Arg1> and indexible<Arg2>, int> = 0>
#endif
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    static_assert(vector_space_descriptors_may_match_with<Arg1, Arg2>, "Dimensions must match");

    if constexpr (has_dynamic_dimensions<Arg1> or has_dynamic_dimensions<Arg2>)
      if (not vector_space_descriptors_match(arg1, arg2))
    {
      auto ret = ::testing::AssertionFailure();
      ret << "Dimensions of first argument (";
      detail::print_dimensions(ret, arg1, std::make_index_sequence<index_count_v<Arg1>>{});
      ret << ") and second argument (";
      detail::print_dimensions(ret, arg2, std::make_index_sequence<index_count_v<Arg2>>{});
      return ret << ") do not match";
    }

    return TestComparison<Arg1, Arg2, Err> {arg1, arg2, err};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err>
#else
  template<typename Arg1, typename Arg2, typename Err, typename>
#endif
  struct TestComparison : ::testing::AssertionResult
  {
    TestComparison(const Arg1& A, const Arg2& B, const Err& err)
      : ::testing::AssertionResult {is_near(to_dense_object(A), to_dense_object(B), err)} {};
  };


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
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err = double>
  requires detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err> and
    (collections::size_of_v<Arg1> == collections::size_of_v<Arg2>) and
    (dynamic_dimension<typename Arg1::value_type, 0> or dynamic_dimension<typename Arg2::value_type, 0> or
      index_dimension_of_v<typename Arg1::value_type, 0> == index_dimension_of_v<typename Arg2::value_type, 0>) and
    (dynamic_dimension<typename Arg1::value_type, 1> or dynamic_dimension<typename Arg2::value_type, 1> or
      index_dimension_of_v<typename Arg1::value_type, 1> == index_dimension_of_v<typename Arg2::value_type, 1>)
#else
  template<typename Arg1, typename Arg2, typename Err = double, std::enable_if_t<
    detail::is_std_array_v<Arg1> and detail::is_std_array_v<Arg2> and std::is_arithmetic_v<Err> and
    (collections::size_of<Arg1>::value == collections::size_of<Arg2>::value) and
    (dynamic_dimension<typename Arg1::value_type, 0> or dynamic_dimension<typename Arg2::value_type, 0> or
      index_dimension_of<typename Arg1::value_type, 0>::value == index_dimension_of<typename Arg2::value_type, 0>::value) and
    (dynamic_dimension<typename Arg1::value_type, 1> or dynamic_dimension<typename Arg2::value_type, 1> or
      index_dimension_of<typename Arg1::value_type, 1>::value == index_dimension_of<typename Arg2::value_type, 1>::value),
      int> = 0>
#endif
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    if constexpr (collections::size_of_v<Arg1> != collections::size_of_v<Arg2>)
    {
      return ::testing::AssertionFailure() << std::endl << "Size of first array (" <<
        collections::size_of_v<Arg1> << " elements) does not match size of second array (" <<
        collections::size_of_v<Arg2> << " elements)" << std::endl;
    }
    else
    {
      return TestComparison<Arg1, Arg2, Err> {arg1, arg2, err};
    }

    if constexpr (dynamic_dimension<std::tuple_element_t<0, Arg1>, 0> or dynamic_dimension<std::tuple_element_t<0, Arg2>, 0>)
      if (get_pattern_collection<0>(std::get<0>(arg1)) != get_pattern_collection<0>(std::get<0>(arg2)))
        throw std::logic_error {"Row dimension mismatch: " + std::to_string(get_pattern_collection<0>(std::get<0>(arg1))) + " != " +
          std::to_string(get_pattern_collection<0>(std::get<0>(arg2))) + " in is_near(array) of " + std::string {__FILE__}};

    if constexpr (dynamic_dimension<std::tuple_element_t<0, Arg1>, 1> or dynamic_dimension<std::tuple_element_t<0, Arg2>, 1>)
      if (get_pattern_collection<1>(std::get<0>(arg1)) != get_pattern_collection<1>(std::get<0>(arg2)))
        throw std::logic_error {"Column dimension mismatch: " + std::to_string(get_pattern_collection<1>(std::get<0>(arg1))) + " != " +
          std::to_string(get_pattern_collection<1>(std::get<0>(arg2))) + " in is_near(array) of " + std::string {__FILE__}};
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
        (dynamic_dimension<Arg1, 0> or dynamic_dimension<Arg2, 0> or index_dimension_of_v<Arg1, 0> == index_dimension_of_v<Arg2, 0>) and
        (dynamic_dimension<Arg1, 1> or dynamic_dimension<Arg2, 1> or index_dimension_of_v<Arg1, 1> == index_dimension_of_v<Arg2, 1>) and
        tuple_sizes_match<std::tuple<Args1...>, std::tuple<Args2...>>::value;
    };

  }


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


}


#endif
