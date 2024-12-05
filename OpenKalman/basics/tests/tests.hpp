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
 * \internal
 * \brief Basic utilities for OpenKalman testing.
 */

#ifndef OPENKALMAN_TESTS_HPP
#define OPENKALMAN_TESTS_HPP

#include <type_traits>
#include <tuple>
#include <string>
#include <gtest/gtest.h>
#include "basics/global-definitions.hpp"


namespace OpenKalman::test
{
  /**
   * \internal
   * \brief Compare two objects.
   * \tparam Arg1 The first object
   * \tparam Arg2 The second object
   * \tparam Err A margin of error
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err>
#else
  template<typename Arg1, typename Arg2, typename Err, typename = void>
#endif
  struct TestComparison;


  /**
   * \internal
   * \brief Deduction guide
   */
  template<typename Arg1, typename Arg2, typename Err>
  TestComparison(Arg1, Arg2, Err) -> TestComparison<std::decay_t<Arg1>, std::decay_t<Arg2>, std::decay_t<Err>>;


  /**
   * \brief Determine if two objects are "near" each other, within a margin of error.
   * \param arg1 The first object
   * \param arg2 The second object
   * \param err The margin of error
   * \return
   */
  template<typename Arg1, typename Arg2, typename Err = double>
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    return TestComparison {arg1, arg2, err};
  }


  /**
   * \brief Compare two tuple-like objects
   * \tparam Arg1 The first tuple-like object
   * \tparam Arg2 The second tuple-like object
   * \tparam Err The margin of error for each element of the tuple-like object. This can, itself, be a tuple.
   */
#ifdef __cpp_concepts
  template<internal::tuple_like Arg1, internal::tuple_like Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
    internal::tuple_like<Arg1> and internal::tuple_like<Arg2>>>
#endif
    : ::testing::AssertionResult
  {
  private:

    template<std::size_t...Ix>
    static ::testing::AssertionResult
    compare(const Arg1 arg1, const Arg2 arg2, const Err& err, std::index_sequence<Ix...>)
    {
      static_assert(std::tuple_size_v<Arg1> == std::tuple_size_v<Arg2>, "tuple size of arguments must match");

      if constexpr (internal::tuple_like<Err>)
      {
        static_assert(std::tuple_size_v<Err> == std::tuple_size_v<Arg1>, "tuple size of error margins must match that of arguments");

        if ((... and OpenKalman::test::TestComparison {std::get<Ix>(arg1), std::get<Ix>(arg2), std::get<Ix>(err)}))
        {
          return ::testing::AssertionSuccess();
        }
        else
        {
          return (::testing::AssertionFailure() << ... << (std::string(Ix == 0 ? "" : ", ") +
            std::string(OpenKalman::test::TestComparison {std::get<Ix>(arg1), std::get<Ix>(arg2), std::get<Ix>(err)} ? "true" : "false")));
        }
      }
      else
      {
        if ((... and OpenKalman::test::TestComparison {std::get<Ix>(arg1), std::get<Ix>(arg2), err}))
        {
          return ::testing::AssertionSuccess();
        }
        else
        {
          return (::testing::AssertionFailure() << ... << (std::string(Ix == 0 ? "" : ", ") +
            std::string(OpenKalman::test::TestComparison {std::get<Ix>(arg1), std::get<Ix>(arg2), err} ? "true" : "false")));
        }
      }
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err, std::make_index_sequence<std::tuple_size_v<Arg1>>{})} {}

  };


} // namespace OpenKalman::test


#endif //OPENKALMAN_TESTS_HPP
