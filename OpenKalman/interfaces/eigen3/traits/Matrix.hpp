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

#ifndef OPENKALMAN_EIGEN3_TRAITS_MATRIX_HPP
#define OPENKALMAN_EIGEN3_TRAITS_MATRIX_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct IndexTraits<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    : detail::IndexTraits_Eigen_default<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>> {};
#endif


  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct Dependencies<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
  {
    static constexpr bool has_runtime_parameters = true;
    using type = std::tuple<>;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_MATRIX_HPP
