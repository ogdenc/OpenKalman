/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for global classes
 */

#include <gtest/gtest.h>
#include "basics/tests/tests.hpp"

using namespace OpenKalman;

#include "basics/classes/equal_to.hpp"

TEST(basics, equal_to)
{
  static_assert(equal_to<int>{}(3, 3));
  static_assert(equal_to<long long>{}(3, 3LL));
  static_assert(equal_to{}(3, 3));
  static_assert(equal_to{}(3, 3u));
  constexpr int i3 = 3;
  constexpr unsigned u3 = 3;
  static_assert(equal_to{}(i3, u3));
  static_assert(equal_to{}(i3, 3LL));
  static_assert(equal_to{}(3, u3));
  static_assert(equal_to{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 5LL}));
}


#include "basics/classes/not_equal_to.hpp"

TEST(basics, not_equal_to)
{
  static_assert(not_equal_to<int>{}(3, 4));
  static_assert(not_equal_to<long long>{}(3, 4LL));
  static_assert(not_equal_to{}(3, 4));
  static_assert(not_equal_to{}(3, 4u));
  static_assert(not_equal_to{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 6LL}));
  static_assert(not not_equal_to{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 5LL}));
}


#include "basics/classes/less.hpp"

TEST(basics, less)
{
  static_assert(less<int>{}(3, 4));
  static_assert(less<long long>{}(3LL, 4));
  static_assert(less{}(3, 4));
  static_assert(less{}(3, 4u));
  static_assert(less{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 6LL}));
  static_assert(not less{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 5LL}));
  static_assert(not less{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 4LL}));
}


#include "basics/classes/less_equal.hpp"

TEST(basics, less_equal)
{
  static_assert(less_equal<int>{}(3, 4));
  static_assert(less_equal<long long>{}(3LL, 3));
  static_assert(less_equal{}(3, 4));
  static_assert(less_equal{}(3, 4u));
  static_assert(less_equal{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 6LL}));
  static_assert(less_equal{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 5LL}));
  static_assert(not less_equal{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 4LL}));
}


#include "basics/classes/greater.hpp"

TEST(basics, greater)
{
  static_assert(greater<int>{}(3, 2));
  static_assert(greater<long long>{}(3, 2LL));
  static_assert(greater{}(3, 2));
  static_assert(greater{}(3, 2u));
  static_assert(greater{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 4LL}));
  static_assert(not greater{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 5LL}));
  static_assert(not greater{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 6LL}));
}


#include "basics/classes/greater_equal.hpp"

TEST(basics, greater_equal)
{
  static_assert(greater_equal<int>{}(3, 2));
  static_assert(greater_equal<long long>{}(3LL, 3));
  static_assert(greater_equal{}(3, 2));
  static_assert(greater_equal{}(3, 2u));
  static_assert(greater_equal{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 4LL}));
  static_assert(greater_equal{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 5LL}));
  static_assert(not greater_equal{}(std::tuple{3, 4, 5}, std::tuple{3LL, 4LL, 6LL}));
}


#include "basics/classes/movable_wrapper.hpp"

TEST(basics, movable_wrapper)
{
  //owning
  constexpr internal::movable_wrapper mr {6};
  static_assert(mr.get() == 6);
  static_assert(internal::movable_wrapper{5}.get() == 5);
  static_assert(static_cast<int>(mr) == 6);
  static_assert(static_cast<int>(std::as_const(mr)) == 6);
  static_assert(internal::movable_wrapper{5} == 5);
  static_assert(internal::movable_wrapper{5} < 6);
  static_assert(internal::movable_wrapper{5} > 4);
  static_assert(5 == internal::movable_wrapper{5});
  static_assert(6 > internal::movable_wrapper{5});
  static_assert(4 < internal::movable_wrapper{5});

  //non-owning
  static constexpr auto i = 7;
  constexpr internal::movable_wrapper ml {i};
  static_assert(ml.get() == 7);
  static_assert(internal::movable_wrapper{i}.get() == 7);
  static_assert(internal::movable_wrapper{std::as_const(i)}.get() == 7);
  static_assert(static_cast<int>(ml) == 7);
  static_assert(static_cast<int>(std::as_const(ml)) == 7);
  static_assert(static_cast<const int&>(ml) == 7);
  static_assert(internal::movable_wrapper{i} == 7);
  static_assert(internal::movable_wrapper{i} < 8);
  static_assert(internal::movable_wrapper{i} > 6);
  static_assert(7 == internal::movable_wrapper{i});
  static_assert(6 < internal::movable_wrapper{i});
  static_assert(8 > internal::movable_wrapper{i});
}

