/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Basic utilities for OpenKalman testing relating to collections.
 */

#ifndef OPENKALMAN_COLLECTIONS_TESTS_HPP
#define OPENKALMAN_COLLECTIONS_TESTS_HPP

#include <tuple>
#include <string>
#include "values/tests/tests.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::test
{
  /**
   * \brief Compare two tuple-like objects
   * \tparam Arg1 The first \ref collections::sized "sized" \ref collections::collection "collection"
   * \tparam Arg2 The second \ref collections::sized "sized" \ref collections::collection "collection"
   * \tparam Err The margin of error for each element of the tuple-like object. This can, itself,
   * be a \ref collections::sized "sized" \ref collections::collection "collection".
   */
#ifdef __cpp_concepts
  template<collections::collection Arg1, collections::collection Arg2, typename Err> requires
    (collections::tuple_like<Arg1> and collections::tuple_like<Arg2> and (collections::tuple_like<Err> or values::value<Err>)) or
    (collections::sized_random_access_range<Arg1> and collections::sized_random_access_range<Arg2> and
      (collections::sized_random_access_range<Err> or values::value<Err>))
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
    collections::collection<Arg1> and collections::collection<Arg2> and
    ((collections::tuple_like<Arg1> and collections::tuple_like<Arg2> and (collections::tuple_like<Err> or values::value<Err>)) or
      (collections::sized_random_access_range<Arg1> and collections::sized_random_access_range<Arg2> and
        (collections::sized_random_access_range<Err> or values::value<Err>)))>>
#endif
    : ::testing::AssertionResult
  {
  private:

    template<std::size_t i, typename Arg>
    static constexpr auto geti(const Arg& arg) { return internal::generalized_std_get<i>(arg); }


    template<std::size_t...Ix>
    static auto
    compare_tuple_like(const Arg1 arg1, const Arg2 arg2, const Err& err, std::index_sequence<Ix...>)
    {
      if constexpr (collections::tuple_like<Err>)
      {
        if ((... and OpenKalman::test::TestComparison {geti<Ix>(arg1), geti<Ix>(arg2), geti<Ix>(err)}))
        {
          return ::testing::AssertionSuccess();
        }
        else
        {
          return (::testing::AssertionFailure() << ... << (std::string(Ix == 0 ? "" : ", ") +
            std::string(OpenKalman::test::TestComparison {geti<Ix>(arg1), geti<Ix>(arg2), geti<Ix>(err)} ?
              std::to_string(geti<Ix>(arg1)) + "==" + std::to_string(geti<Ix>(arg2)) :
              std::to_string(geti<Ix>(arg1)) + "!=" + std::to_string(geti<Ix>(arg2)) )));
        }
      }
      else
      {
        if ((... and OpenKalman::test::TestComparison {geti<Ix>(arg1), geti<Ix>(arg2), err}))
        {
          return ::testing::AssertionSuccess();
        }
        else
        {
          return (::testing::AssertionFailure() << ... << (std::string(Ix == 0 ? "" : ", ") +
            std::string(OpenKalman::test::TestComparison {geti<Ix>(arg1), geti<Ix>(arg2), err} ?
              std::to_string(geti<Ix>(arg1)) + " == " + std::to_string(geti<Ix>(arg2)) :
              std::to_string(geti<Ix>(arg1)) + " != " + std::to_string(geti<Ix>(arg2)) )));
        }
      }
    }


    template<std::size_t...Ix>
    static auto
    compare(const Arg1 arg1, const Arg2 arg2, const Err& err)
    {
      if constexpr (collections::size_of_v<Arg1> != dynamic_size and collections::size_of_v<Arg2> != dynamic_size)
        static_assert(collections::size_of_v<Arg1> == collections::size_of_v<Arg2>, "size of arguments must match");

      if constexpr (collections::collection<Err>)
        if constexpr (collections::size_of_v<Err> != dynamic_size and collections::size_of_v<Arg1> != dynamic_size)
          static_assert(collections::size_of_v<Err> == collections::size_of_v<Arg1>, "size of error margins must match that of arguments");

      if constexpr (collections::tuple_like<Arg1> and collections::tuple_like<Arg2>)
      {
        return compare_tuple_like(arg1, arg2, err, std::make_index_sequence<std::tuple_size_v<Arg1>>{});
      }
      else // if constexpr (collections::sized_random_access_range<Arg1> and collections::sized_random_access_range<Arg2>)
      {
        std::string message;
        bool success = true;
        auto it1 = arg1.begin();
        auto it2 = arg2.begin();
        std::size_t count = 0;
        if constexpr (collections::sized_random_access_range<Err>)
        {
          for (auto ite = err.begin(); it1 != arg1.end() and it2 != arg2.end() and ite != err.end(); ++it1, ++it2, ++ite, ++count)
          {
            if (not OpenKalman::test::TestComparison {*it1, *it2, *ite})
            {
              success = false;
              message += std::to_string(count) + ": " + std::to_string(*it1) + "!=" + std::to_string(*it2) + ". ";
            }
          }
        }
        else
        {
          for (; it1 != arg1.end() and it2 != arg2.end(); ++it1, ++it2, ++count)
          {
            if (not OpenKalman::test::TestComparison {*it1, *it2, err})
            {
              success = false;
              message += std::to_string(count) + ": " + std::to_string(*it1) + "!=" + std::to_string(*it2) + ". ";
            }
          }
        }
        if (success) return ::testing::AssertionSuccess();
        else return ::testing::AssertionFailure() << message;
      }
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err)} {}

  };

}


#endif
