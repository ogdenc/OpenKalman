/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEBASEBASE_H
#define OPENKALMAN_COVARIANCEBASEBASE_H

namespace OpenKalman::internal
{
  /**
   * Ultimate base of Covariance and SquareRootCovariance classes.
   * \TODO: combine with OpenKalman::internal::MatrixBase
   */
  template<typename Derived, typename ArgType>
  struct CovarianceBaseBase : MatrixTraits<ArgType>::template MatrixBaseType<Derived>
  {
  };


}

#endif //OPENKALMAN_COVARIANCEBASEBASE_H
