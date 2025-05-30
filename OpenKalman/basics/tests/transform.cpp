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
 * \brief Tests for \ref ranges::transform_view and \ref ranges::views::transform.
 */

#include <vector>
#include "tests.hpp"
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/iota.hpp"
#include "basics/compatibility/views/transform.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif
namespace vw = rg::views;

TEST(basics, transform_view)
{
  EXPECT_EQ(rg::transform_view(rg::iota_view(0u, 5u), [](auto i){ return i; })[3u], 3);
  static_assert(rg::transform_view(rg::iota_view(0u, 5u), [](auto i){ return i; }).size() == 5u);
  auto r_identity = rg::transform_view(rg::iota_view(0u, 5u), [](auto i){ return i; });
  EXPECT_EQ(r_identity[3u], 3);
  EXPECT_EQ(r_identity.size(), 5_uz);
  std::size_t j = 0;
  for (auto i : r_identity) EXPECT_EQ(i, j++);

  auto r_reverse = rg::transform_view(rg::iota_view(0u, 9u), [](auto i){ return 10u - i; });
  EXPECT_EQ(r_reverse.size(), 9u);
  j = 10;
  for (auto i : r_reverse) EXPECT_EQ(i, j--);

  auto ita = r_reverse.begin();
  EXPECT_EQ(*ita, 10);
  EXPECT_EQ(*(ita + 1), 9);
  EXPECT_EQ(ita[1], 9);
  EXPECT_EQ(*(2 + ita), 8);
  EXPECT_EQ(ita[3], 7);
  ++ita;
  EXPECT_EQ(*ita, 9);
  EXPECT_EQ(*(ita - 1), 10);
  EXPECT_EQ(ita[-1], 10);
  EXPECT_EQ(ita[1], 8);
  EXPECT_EQ(ita++[2], 7);
  EXPECT_EQ(*ita, 8);
  EXPECT_EQ(*(ita - 2), 10);
  EXPECT_EQ(ita[1], 7);
  EXPECT_EQ(ita--[2], 6);
  EXPECT_EQ(*ita, 9);
  --ita;
  EXPECT_EQ(*ita, 10);

  EXPECT_EQ((vw::iota(0u, 5u) | vw::transform([](auto i){ return i + 3u; }))[0u], 3u);
  EXPECT_EQ((vw::iota(0u, 5u) | vw::transform([](auto i){ return i * 2u; }))[3u], 6u);
}
