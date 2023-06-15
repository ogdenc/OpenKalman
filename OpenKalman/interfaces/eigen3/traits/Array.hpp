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
 * \brief Type traits as applied to Eigen::Array.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_ARRAY_HPP
#define OPENKALMAN_EIGEN3_TRAITS_ARRAY_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  struct IndexTraits<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : detail::IndexTraits_Eigen_default<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {};
#endif


  template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  struct Dependencies<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
  {
    static constexpr bool has_runtime_parameters = true;
    using type = std::tuple<>;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_ARRAY_HPP
