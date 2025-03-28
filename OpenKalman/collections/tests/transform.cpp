/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref collections::transform_view and \ref collections::views::transform.
 */

#include "values/tests/tests.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/dynamic.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/concepts/index.hpp"
#include "values/functions/internal/get_collection_element.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "values/functions/iota_collection.hpp"

#include "values/functions/transform_collection.hpp"

TEST(values, transform_collection)
{
  using value::transform_collection;
  auto c0 = std::integral_constant<std::size_t, 0>{};
  auto c5 = std::integral_constant<std::size_t, 5>{};
  constexpr auto t_identity = transform_collection(value::iota_collection(c0, c5), [](auto i){ return i; });
  static_assert(std::tuple_size_v<std::decay_t<decltype(t_identity)>> == 5);
  using value::get;
  static_assert(get<3>(t_identity) == 3);

  auto r_identity = transform_collection(value::iota_collection(0u, 5u), [](auto i){ return i; });
  EXPECT_EQ(r_identity.size(), 5_uz);
  std::size_t j = 0;
  for (auto i : r_identity) EXPECT_EQ(i, j++);

  auto r_reverse = transform_collection(value::iota_collection(0u, 9u), [](auto i){ return 10_uz - i; });
  EXPECT_EQ(r_reverse.size(), 9_uz);
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
}

