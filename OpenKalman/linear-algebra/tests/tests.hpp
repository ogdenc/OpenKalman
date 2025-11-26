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

#include "collections/collections.hpp"
#include "collections/tests/tests.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_index_extent.hpp"

namespace OpenKalman::test
{
  // ------------------- //
  //  indexible objects  //
  // ------------------- //

#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2, typename Err> requires
    (index_count_v<Arg1> == index_count_v<Arg2>) and
    (values::value<Err> or (indexible<Err> and index_count_v<Err> == index_count_v<Arg1>)) and
    (not collections::collection<Arg1> or not collections::collection<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
    template<typename Arg1, typename Arg2, typename Err>
    struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
      indexible<Arg1> and indexible<Arg2> and
      (index_count<Arg1>::value == index_count<Arg2>::value) and
      (values::value<Err> or indexible<Err>) and
      (not collections::collection<Arg1> or not collections::collection<Arg2>)>>
#endif
    : ::testing::AssertionResult
  {
  private:

    static constexpr auto
    print_indices() { return std::string{"()"}; }

    template<typename I, typename...Is>
    static constexpr auto
    print_indices(I i, Is...is) { return std::string{"("} + (std::to_string(i) + ... + (", " + std::to_string(is))) + ")"; }


    template<typename...Ix>
    static ::testing::AssertionResult
    compare_mdspan(const Arg1& arg1, const Arg2& arg2, const Err& err, Ix...ix)
    {
      constexpr std::size_t ind = sizeof...(Ix);
      if constexpr (ind < index_count_v<Arg1>)
      {
        auto dim1 = get_index_extent<ind>(arg1);
        auto dim2 = get_index_extent<ind>(arg2);
        if (dim1 != dim2) return ::testing::AssertionFailure() << "Dimensions do not match for index" <<
          std::to_string(ind) << " (" << std::to_string(dim1) << " != " << std::to_string(dim2) << ")";
        std::string msg = "";
        for (std::size_t i = 0; i < get_index_extent<ind>(arg1); ++i)
          msg += compare_mdspan(arg1, arg2, err, ix..., i).message();
        if (msg.size() == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << msg;
      }
      else
      {
        auto indices = std::array{ix...};
        auto e = [&]{ if constexpr (indexible<Err>) return err[indices]; else return err; }();
        auto res = test::TestComparison {get_mdspan(arg1)[indices], get_mdspan(arg2)[indices], e};
        if (res) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << print_indices(ix...) << ": " << res.message() << std::endl;
      }
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare_mdspan(arg1, arg2, err)} {};

  };


}


#endif
