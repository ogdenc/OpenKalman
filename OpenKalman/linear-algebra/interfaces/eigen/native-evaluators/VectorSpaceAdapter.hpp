/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 evaluator for VectorSpaceAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_VECTORSPACEADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_VECTORSPACEADAPTER_HPP

namespace Eigen::internal
{
  template<typename NestedObject, typename...Vs>
  struct evaluator<OpenKalman::VectorSpaceAdapter<NestedObject, Vs...>>
    : evaluator<std::decay_t<NestedObject>>
  {
    using XprType = OpenKalman::VectorSpaceAdapter<NestedObject, Vs...>;
    using Base = evaluator<std::decay_t<NestedObject>>;
    explicit evaluator(const XprType& arg) : Base {OpenKalman::nested_object(arg)} {}

    enum {
      Flags = (Base::Flags & ~Eigen::RowMajorBit) | (traits<XprType>::Flags & RowMajorBit),
    };

  };

}

#endif
