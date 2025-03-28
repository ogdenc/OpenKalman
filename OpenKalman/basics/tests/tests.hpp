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
 * \internal
 * \brief Basic utilities for OpenKalman testing.
 */

#ifndef OPENKALMAN_TESTS_HPP
#define OPENKALMAN_TESTS_HPP

#include <type_traits>
#include <tuple>
#include <string>
#include <gtest/gtest.h>
#include "collections/concepts/tuple_like.hpp"


namespace OpenKalman::test
{
  /**
   * \internal
   * \brief Compare two objects.
   * \tparam Arg1 The first object
   * \tparam Arg2 The second object
   * \tparam Err A margin of error
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2, typename Err>
#else
  template<typename Arg1, typename Arg2, typename Err, typename = void>
#endif
  struct TestComparison;


  /**
   * \internal
   * \brief Deduction guide
   */
  template<typename Arg1, typename Arg2, typename Err>
  TestComparison(const Arg1&, const Arg2&, const Err&) -> TestComparison<Arg1, Arg2, Err>;


  /**
   * \brief Determine if two objects are "near" each other, within a margin of error.
   * \param arg1 The first object
   * \param arg2 The second object
   * \param err The margin of error
   * \return
   */
  template<typename Arg1, typename Arg2, typename Err = double>
  inline ::testing::AssertionResult is_near(const Arg1& arg1, const Arg2& arg2, const Err& err = 1e-6)
  {
    return TestComparison {arg1, arg2, err};
  }

} // namespace OpenKalman::test


#endif //OPENKALMAN_TESTS_HPP
