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

#ifndef OPENKALMAN_EIGEN_TRAITS_ARRAY_HPP
#define OPENKALMAN_EIGEN_TRAITS_ARRAY_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  struct object_traits<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : Eigen3::object_traits_base<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
  {
  private:

    using Base = Eigen3::object_traits_base<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>;

  public:

    // nested_object() not defined

    // get_constant not defined

    // get_constant_diagonal not defined

    static constexpr data_layout layout = Base::row_major ? data_layout::right : data_layout::left;

  };

}

#endif
