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
 * \file transforms.hpp
 * A meta-header file including all the headers relating to OpenKalman transforms.
 */

#ifndef OPENKALMAN_TRANSFORMS_HPP
#define OPENKALMAN_TRANSFORMS_HPP

#include "details/ScaledSigmaPointsBase.hpp"
#include "sample-points/sigma-points/Unscented.hpp"
#include "sample-points/sigma-points/SphericalSimplex.hpp"
#include "sample-points/sigma-points/SigmaPoints.hpp"
#include "sample-points/CubaturePoints.hpp"

#include "details/TransformBase.hpp"
#include "details/LinearTransformBase.hpp"
#include "LinearTransform.hpp"
#include "LinearizedTransform.hpp"
#include "sample-points/SamplePointsTransform.hpp"
#include "MonteCarloTransform.hpp"
#include "IdentityTransform.hpp"
#include "RecursiveLeastSquaresTransform.hpp"

#endif //OPENKALMAN_TRANSFORMS_HPP
