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
 * \brief Tests for \ref ranges::empty_view and \ref ranges::views::empty
 */

#include "tests.hpp"
#include "basics/compatibility/views/empty.hpp"

using namespace OpenKalman;

TEST(basics, empty_view)
{
  static_assert(stdex::ranges::view<stdex::ranges::empty_view<int>>);
  static_assert(stdex::ranges::viewable_range<stdex::ranges::empty_view<double>>);
  static constexpr auto e1 = stdex::ranges::views::empty<int>;
  static_assert(e1.begin() == nullptr);
  static_assert(e1.end() == nullptr);
  static_assert(stdex::ranges::begin(e1) == nullptr);
  static_assert(stdex::ranges::end(e1) == nullptr);
  static_assert(e1.data() == nullptr);
  static_assert(e1.empty());
}


