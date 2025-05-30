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
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/empty.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif
namespace vw = rg::views;

TEST(basics, empty_view)
{
  static_assert(rg::view<rg::empty_view<int>>);
  static_assert(rg::viewable_range<rg::empty_view<double>>);
  static constexpr auto e1 = vw::empty<int>;
  static_assert(e1.begin() == nullptr);
  static_assert(e1.end() == nullptr);
  static_assert(rg::begin(e1) == nullptr);
  static_assert(rg::end(e1) == nullptr);
  static_assert(e1.data() == nullptr);
  static_assert(e1.empty());
}


