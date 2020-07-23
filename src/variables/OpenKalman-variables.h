/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_OPENKALMAN_VARIABLES_H
#define OPENKALMAN_OPENKALMAN_VARIABLES_H

#include "variables/support/Traits.h"
#include "variables/support/ForwardDeclarations.h"
#include "variables/support/OpenKalman-coefficients.h"
#include "variables/support/ElementSetter.h"
//
#include "variables/support/MatrixBase.h"
#include "variables/support/TypedMatrixBase.h"
#include "variables/classes/Mean.h"
#include "variables/classes/EuclideanMean.h"
#include "variables/classes/TypedMatrix.h"
#include "variables/support/TypedMatrixOverloads.h"
//
#include "variables/support/ConvertBaseMatrix.h"
#include "variables/support/CovarianceBaseBase.h"
#include "variables/support/CovarianceBase.h"
#include "variables/classes/Covariance.h"
#include "variables/classes/SquareRootCovariance.h"
#include "variables/support/CovarianceOverloads.h"

#endif //OPENKALMAN_OPENKALMAN_VARIABLES_H
