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
  static_assert(stdcompat::ranges::view<stdcompat::ranges::empty_view<int>>);
  static_assert(stdcompat::ranges::viewable_range<stdcompat::ranges::empty_view<double>>);
  static constexpr auto e1 = stdcompat::ranges::views::empty<int>;
  static_assert(e1.begin() == nullptr);
  static_assert(e1.end() == nullptr);
  static_assert(stdcompat::ranges::begin(e1) == nullptr);
  static_assert(stdcompat::ranges::end(e1) == nullptr);
  static_assert(e1.data() == nullptr);
  static_assert(e1.empty());
}


