/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for language-features.hpp
 */

#include <gtest/gtest.h>
#include "basics/language-features.hpp"


TEST(basics, uz_literal)
{
  static_assert(std::is_same_v<decltype(5_uz), std::size_t>);
}


#ifndef __cpp_lib_remove_cvref
TEST(basics, remove_cvref)
{
  static_assert(std::is_same_v<OpenKalman::remove_cvref_t<int[5]>, int[5]>);
  static_assert(std::is_same_v<OpenKalman::remove_cvref_t<const int[5]>, int[5]>);
}
#endif
