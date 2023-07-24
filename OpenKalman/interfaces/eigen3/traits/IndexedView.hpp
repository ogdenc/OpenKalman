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

#ifndef OPENKALMAN_EIGEN3_TRAITS_INDEXEDVIEW_HPP
#define OPENKALMAN_EIGEN3_TRAITS_INDEXEDVIEW_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  // \todo: Add other interface traits.

  template<typename XprType, typename RowIndices, typename ColIndices>
  struct IndexibleObjectTraits<Eigen::IndexedView<XprType, RowIndices, ColIndices>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::IndexedView<XprType, RowIndices, ColIndices>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::IndexedView<XprType, RowIndices, ColIndices>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_INDEXEDVIEW_HPP
