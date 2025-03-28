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
 * \brief Tests for global-definitions.hpp
 */

#include <gtest/gtest.h>
#include <complex>
#include "basics/global-definitions.hpp"

using namespace OpenKalman;


TEST(basics, Applicability)
{

  static_assert(Applicability::guaranteed == not Applicability::permitted);
  static_assert((not Applicability::guaranteed) == Applicability::permitted);

  static_assert((Applicability::guaranteed and Applicability::guaranteed) == Applicability::guaranteed);
  static_assert((Applicability::guaranteed and Applicability::permitted) == Applicability::permitted);
  static_assert((Applicability::permitted and Applicability::guaranteed) == Applicability::permitted);
  static_assert((Applicability::permitted and Applicability::permitted) == Applicability::permitted);

  static_assert((Applicability::guaranteed or Applicability::guaranteed) == Applicability::guaranteed);
  static_assert((Applicability::guaranteed or Applicability::permitted) == Applicability::guaranteed);
  static_assert((Applicability::permitted or Applicability::guaranteed) == Applicability::guaranteed);
  static_assert((Applicability::permitted or Applicability::permitted) == Applicability::permitted);
}


TEST(basics, remove_rvalue_reference)
{
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<double&&>, double>);
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<const double&&>, const double>);
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<double&>, double&>);
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<const double>, const double>);
}

