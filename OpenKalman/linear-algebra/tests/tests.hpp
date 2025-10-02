/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "linear-algebra/concepts/has_dynamic_dimensions.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/vector_space_descriptors_may_match_with.hpp"
#include "../traits/get_pattern_collection.hpp"
#include "linear-algebra/functions/n_ary_operation.hpp"
#include "linear-algebra/functions/reduce.hpp"
#include "linear-algebra/traits/index_count.hpp"

namespace OpenKalman::test
{
  // ------------------- //
  //  indexible objects  //
  // ------------------- //

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
      return (res << ... << (std::to_string(get_index_extent<Is>(arg)) + (Is + 1 < sizeof...(Is) ? "," : "")));
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
          print_dimensions(ret, arg1, std::make_index_sequence<index_count_v<Arg1>>{});
          ret << ") and second argument (";
          print_dimensions(ret, arg2, std::make_index_sequence<index_count_v<Arg2>>{});
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


}


#endif
