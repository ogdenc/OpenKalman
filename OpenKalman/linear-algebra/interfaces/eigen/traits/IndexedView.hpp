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
 * \brief Type traits as applied to Eigen::IndexedView (Eigen 3.4).
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_INDEXEDVIEW_HPP
#define OPENKALMAN_EIGEN_TRAITS_INDEXEDVIEW_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  // \todo: Add other interface traits.

  template<typename XprType, typename RowIndices, typename ColIndices>
  struct object_traits<Eigen::IndexedView<XprType, RowIndices, ColIndices>>
    : Eigen3::object_traits_base<Eigen::IndexedView<XprType, RowIndices, ColIndices>>
  {
  };

}

#endif
