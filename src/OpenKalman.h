/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * @file
 * This is the meta-header file for OpenKalman.
 *
 * Include this file before using any OpenKalman classes or methods.
 * It should appear *after* an interface header file (e.g., Eigen3.h), for the matrix library.
 */

#ifndef OPENKALMAN_OPENKALMAN_H
#define OPENKALMAN_OPENKALMAN_H

/**
 * The namespace for all OpenKalman-specific classes and methods.
 */
namespace OpenKalman
{}

#include "variables/OpenKalman-variables.h"
#include "distributions/GaussianDistribution.h"
#include "transforms/OpenKalman-transforms.h"
#include "filters/KalmanFilter.h"


#endif //OPENKALMAN_OPENKALMAN_H
