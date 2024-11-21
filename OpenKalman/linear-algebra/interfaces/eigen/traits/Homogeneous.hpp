/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::Homogeneous.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_HOMOGENEOUS_HPP
#define OPENKALMAN_EIGEN_TRAITS_HOMOGENEOUS_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  // \todo: Add other interface traits. This is a child of Eigen::MatrixBase

  template<typename MatrixType, int Direction>
  struct indexible_object_traits<Eigen::Homogeneous<MatrixType, Direction>>
    : Eigen3::indexible_object_traits_base<Eigen::Homogeneous<MatrixType, Direction>>
  {
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_HOMOGENEOUS_HPP
