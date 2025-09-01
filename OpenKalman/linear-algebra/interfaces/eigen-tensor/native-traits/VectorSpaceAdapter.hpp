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
 * \brief Native Eigen3 tensor traits for Eigen3 general VectorSpaceAdapter
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_VECTORSPACEADAPTER_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_VECTORSPACEADAPTER_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_tensor_general NestedObject, typename...Vs>
  struct native_traits<OpenKalman::VectorSpaceAdapter<NestedObject, Vs...>>
#else
  template<typename NestedObject, typename...Vs>
  struct native_traits<OpenKalman::VectorSpaceAdapter<NestedObject, Vs...>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_tensor_general<NestedObject>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedObject>> {};

}


#endif
