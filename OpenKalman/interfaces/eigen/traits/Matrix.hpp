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
 * \brief Type traits as applied to Eigen::Matrix.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_MATRIX_HPP
#define OPENKALMAN_EIGEN_TRAITS_MATRIX_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct IndexibleObjectTraits<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters = true;

    using type = std::tuple<>;

    // get_nested_matrix() not defined

    // convert_to_self_contained() not defined

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_MATRIX_HPP
