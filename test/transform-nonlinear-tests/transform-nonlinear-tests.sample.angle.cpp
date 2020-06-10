/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear-tests.h"

TEST_F(transform_tests, Transform_angle)
{
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot));
}

TEST_F(transform_tests, Transform_Cholesky_angle)
{
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
}
