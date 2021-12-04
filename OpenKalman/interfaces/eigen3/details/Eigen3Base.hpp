/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for Eigen3::Eigen3Base
 * \todo Specialize for Matrix, Covariance, etc., so that they do not derive from Eigen::MatrixBase?
 */

#ifndef OPENKALMAN_EIGEN3BASE_HPP
#define OPENKALMAN_EIGEN3BASE_HPP

namespace OpenKalman::Eigen3::internal
{
  template<typename Derived>
  struct Eigen3Base : Eigen::MatrixBase<Derived>
  {
    /// \internal \note Required by Eigen 3 for this to be used in an Eigen::CwiseBinaryOp.
    using Nested = Eigen3Base;


  };

} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN3BASE_HPP
