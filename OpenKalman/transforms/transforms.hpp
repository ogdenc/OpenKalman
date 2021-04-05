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
 * \brief Files relating to transforms.
 *
 * \dir transforms/details
 * \brief Support files for transforms.
 *
 * \dir sample-points
 * \brief Files relating to alternative sample point schemes for use with SamplePointsTransform.
 *
 * \file
 * A meta-header file including all the headers relating to OpenKalman transforms.
 */

#ifndef OPENKALMAN_TRANSFORMS_HPP
#define OPENKALMAN_TRANSFORMS_HPP


#include "details/TransformBase.hpp"

#include "IdentityTransform.hpp"

#include "details/LinearTransformBase.hpp"

#include "LinearTransform.hpp"
#include "LinearizedTransform.hpp"

#include "details/ScaledSigmaPointsBase.hpp"

#include "sample-points/Unscented.hpp"
#include "sample-points/SphericalSimplex.hpp"
#include "sample-points/CubaturePoints.hpp"

#include "SamplePointsTransform.hpp"

#include "MonteCarloTransform.hpp"

#include "RecursiveLeastSquaresTransform.hpp"


#endif //OPENKALMAN_TRANSFORMS_HPP
