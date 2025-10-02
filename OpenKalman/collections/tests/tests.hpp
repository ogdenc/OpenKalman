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

#include <string>
#include "values/tests/tests.hpp"
#include "values/values.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/uniformly_gettable.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::test
{
  /**
   * \brief Compare two \ref collection objects
   * \tparam Arg1 The first \ref collections::sized "sized" \ref collections::collection "collection"
   * \tparam Arg2 The second \ref collections::sized "sized" \ref collections::collection "collection"
   * \tparam Err The margin of error for each element of the tuple-like object.
   * This can, itself, be a \ref collections::collection "collection".
   */
#ifdef __cpp_concepts
  template<collections::collection Arg1, collections::collection Arg2, typename Err> requires
    (collections::uniformly_gettable<Arg1> and collections::uniformly_gettable<Arg2> and (collections::uniformly_gettable<Err> or values::value<Err>)) or
    (stdcompat::ranges::range<Arg1> and stdcompat::ranges::range<Arg2> and
      (stdcompat::ranges::range<Err> or values::value<Err>))
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
    collections::collection<Arg1> and collections::collection<Arg2> and
    ((collections::uniformly_gettable<Arg1> and collections::uniformly_gettable<Arg2> and (collections::uniformly_gettable<Err> or values::value<Err>)) or
      (stdcompat::ranges::range<Arg1> and stdcompat::ranges::range<Arg2> and
        (stdcompat::ranges::range<Err> or values::value<Err>)))>>
#endif
    : ::testing::AssertionResult
  {
  private:

    template<std::size_t i, typename Arg>
    static constexpr auto geti(const Arg& arg) { return internal::generalized_std_get<i>(arg); }


    template<std::size_t...Ix>
    static auto
    compare_tuple_like(const Arg1& arg1, const Arg2& arg2, const Err& err, std::index_sequence<Ix...>)
    {
      if constexpr (collections::uniformly_gettable<Err>)
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


    template<typename A1, typename A2, typename E, typename...Ix>
    static bool
    compare_mdarray(const A1& a1, const A2& a2, const E& e, std::string& message, Ix...ix)
    {
      static_assert(std::rank_v<A1> == std::rank_v<A2>);
      if constexpr (std::rank_v<A1> > 0)
      {
        static_assert(std::extent_v<A1> == std::extent_v<A2>, "extents of arguments must match");
        static_assert(not std::is_array_v<Err> or std::extent_v<E> == std::extent_v<A1>, "extents of error argument must match other arguments");
        bool success = true;
        for (std::size_t i = 0; i < std::extent_v<A1>; ++i)
        {
          if constexpr (std::is_array_v<Err>)
          {
            if (not compare_mdarray(a1[i], a2[i], e[i], message, ix..., i)) success = false;
          }
          else
          {
            if (not compare_mdarray(a1[i], a2[i], e, message, ix..., i)) success = false;
          }
        }
        return success;
      }
      else
      {
        if (OpenKalman::test::TestComparison {a1, a2, e}) return true;
        message += ("" + ... + (" " + std::to_string(ix))) + ": " + std::to_string(a1) +
          "!=" + std::to_string(a2) + "(±" + std::to_string(e) +  "). ";
        return false;
      }
    }


    template<std::size_t...Ix>
    static auto
    compare(const Arg1& arg1, const Arg2& arg2, const Err& err)
    {
      static_assert(values::size_compares_with<collections::size_of<Arg1>, collections::size_of<Arg2>,
        &stdcompat::is_eq, applicability::permitted>, "size of arguments must match");

      static_assert(values::value<Err> or values::size_compares_with<collections::size_of<Err>, collections::size_of<Arg1>,
        &stdcompat::is_eq, applicability::permitted>, "size of error margins must match that of arguments");

      if constexpr (collections::uniformly_gettable<Arg1> and collections::uniformly_gettable<Arg2>)
      {
        return compare_tuple_like(arg1, arg2, err, std::make_index_sequence<collections::size_of_v<Arg1>>{});
      }
      else if constexpr (std::rank_v<Arg1> > 1 and std::rank_v<Arg1> == std::rank_v<Arg2> and (std::rank_v<Err> == std::rank_v<Arg1> or values::value<Err>))
      {
        std::string message;
        if (compare_mdarray(arg1, arg2, err, message)) return ::testing::AssertionSuccess();
        else return ::testing::AssertionFailure() << message;
      }
      else // if constexpr (stdcompat::ranges::range<Arg1> and stdcompat::ranges::range<Arg2>)
      {
        std::string message;
        bool success = true;
        auto it1 = stdcompat::ranges::begin(arg1);
        auto it2 = stdcompat::ranges::begin(arg2);
        std::size_t count = 0;
        if constexpr (stdcompat::ranges::range<Err>)
        {
          for (auto ite = stdcompat::ranges::begin(err);
               it1 != stdcompat::ranges::end(arg1) and it2 != stdcompat::ranges::end(arg2) and ite != stdcompat::ranges::end(err);
               ++it1, ++it2, ++ite, ++count)
          {
            if (not OpenKalman::test::TestComparison {*it1, *it2, *ite})
            {
              success = false;
              message += std::to_string(count) + ": " + std::to_string(*it1) + "!=" + std::to_string(*it2) +
                "(±" + std::to_string(*ite) +  "). ";
            }
          }
        }
        else
        {
          for (; it1 != stdcompat::ranges::end(arg1) and it2 != stdcompat::ranges::end(arg2); ++it1, ++it2, ++count)
          {
            if (not OpenKalman::test::TestComparison {*it1, *it2, err})
            {
              success = false;
              message += std::to_string(count) + ": " + std::to_string(*it1) + "!=" + std::to_string(*it2) +
                "(±" + std::to_string(err) +  "). ";
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
