/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to coefficient types
 *
 * \dir coefficient-types/details
 * \brief Files implementing details regarding coefficient types.
 *
 * \dir coefficient-types/tests
 * \brief Test files for coefficient types.
 *
 * \file
 * \brief Comprehensive header file including all coefficient-related classes and definitions
 */

#ifndef OPENKALMAN_COEFFICIENT_TYPES_HPP
#define OPENKALMAN_COEFFICIENT_TYPES_HPP

#include "details/coefficient_forward-declarations.hpp"

#include "Axis.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Inclination.hpp"
#include "Polar.hpp"
#include "Spherical.hpp"
#include "Coefficients.hpp"

#include "details/coefficient-functions.hpp"

#endif //OPENKALMAN_COEFFICIENT_TYPES_HPP
