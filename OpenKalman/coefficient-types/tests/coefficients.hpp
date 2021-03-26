/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Coefficient tests
 *
 * \file
 * \brief Header file for coefficient tests
 */

#include <gtest/gtest.h>

#include "basics/basics.hpp"
#include "coefficient-types/coefficient-types.hpp"

using namespace OpenKalman;


struct coefficients : public ::testing::Test
{
  coefficients() {}

  void SetUp() override {}

  void TearDown() override {}

  ~coefficients() override {}
};

