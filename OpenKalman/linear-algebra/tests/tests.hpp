/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2026 Christopher Lee Ogden <ogden@gatech.edu>
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
    (values::value<Err> or indexible<Err>) and
    (not collections::collection<Arg1> or not collections::collection<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
    template<typename Arg1, typename Arg2, typename Err>
    struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<
      indexible<Arg1> and indexible<Arg2> and
      (values::value<Err> or indexible<Err>) and
      (not collections::collection<Arg1> or not collections::collection<Arg2>)>>
#endif
    : ::testing::AssertionResult
  {
  private:

    static_assert(not indexible<Err> or
      values::fixed_value_compares_with<index_count<Err>, std::max(index_count_v<Arg1>, index_count_v<Arg2>)>);

    static constexpr std::size_t
    rank1 = std::decay_t<decltype(get_mdspan(std::declval<Arg1&>()))>::rank();

    static constexpr std::size_t
    rank2 = std::decay_t<decltype(get_mdspan(std::declval<Arg2&>()))>::rank();


    template<typename Indices>
    static constexpr auto
    print_indices(const Indices& indices, std::index_sequence<>) { return std::string{"()"}; }

    template<typename Indices, std::size_t i, std::size_t...is>
    static constexpr auto
    print_indices(const Indices& indices, std::index_sequence<i, is...>)
    {
      return std::string{"("} + (std::to_string(std::get<i>(indices)) + ... + (", " + std::to_string(std::get<is>(indices)))) + ")";
    }


    template<typename Indices, std::size_t...i1, std::size_t...i2>
    static ::testing::AssertionResult
    compare_element(const Arg1& arg1, const Arg2& arg2, const Err& e, const Indices& indices,
      std::index_sequence<i1...>, std::index_sequence<i2...>)
    {
      auto indices1 = std::array {std::get<i1>(indices)...};
      auto indices2 = std::array {std::get<i2>(indices)...};
      auto res = test::TestComparison {get_mdspan(arg1)[indices1], get_mdspan(arg2)[indices2], e};
      if (res) return ::testing::AssertionSuccess();
      constexpr auto seq = std::make_index_sequence<std::max(rank1, rank2)>{};
      return ::testing::AssertionFailure() << print_indices(indices, seq) << ": " << res.message() << std::endl;
    }


    template<typename...Ix>
    static ::testing::AssertionResult
    compare_mdspan(const Arg1& arg1, const Arg2& arg2, const Err& err, Ix...ix)
    {
      constexpr std::size_t ind = sizeof...(Ix);
      if constexpr (ind < std::min(rank1, rank2))
      {
        auto dim1 = get_index_extent<ind>(arg1);
        auto dim2 = get_index_extent<ind>(arg2);
        if (dim1 != dim2) return ::testing::AssertionFailure() << "Dimensions do not match for index " <<
          std::to_string(ind) << ": " << std::to_string(dim1) << " != " << std::to_string(dim2) << ")";
        std::string msg = "";
        for (std::size_t i = 0; i < dim1; ++i) msg += compare_mdspan(arg1, arg2, err, ix..., i).message();
        if (msg.size() == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << msg;
      }
      else if constexpr (ind < std::max(rank1, rank2))
      {
        auto dim = rank2 > rank1 ? get_index_extent<ind>(arg2) : get_index_extent<ind>(arg1);
        if (dim != 1) return ::testing::AssertionFailure() << "Dimensions do not match for index" <<
          std::to_string(ind) << " (" << (rank2 > rank1 ? "1" : std::to_string(dim)) << " != " << (rank2 < rank1 ? "1" : std::to_string(dim)) << ")";
        std::string msg = compare_mdspan(arg1, arg2, err, ix..., 0_uz).message();
        if (msg.size() == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << msg;
      }
      else
      {
        auto indices = std::array{ix...};
        auto e = [&]{ if constexpr (indexible<Err>) return err[indices]; else return err; }();
        return compare_element(arg1, arg2, e, indices, std::make_index_sequence<rank1>{}, std::make_index_sequence<rank2>{});
      }
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare_mdspan(arg1, arg2, err)} {};

  };


}


#endif
