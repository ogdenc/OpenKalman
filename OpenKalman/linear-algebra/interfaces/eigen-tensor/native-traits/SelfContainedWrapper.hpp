/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen traits relating to \ref SelfContainedWrapper holding a tensor
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SELFCONTAINEDWRAPPER_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SELFCONTAINEDWRAPPER_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_tensor_general T, typename...Ps>
  struct native_traits<OpenKalman::internal::SelfContainedWrapper<T, Ps...>>
#else
  template<typename T, typename...Ps>
  struct native_traits<OpenKalman::internal::SelfContainedWrapper<T, Ps...>, std::enable_if_t<OpenKalman::Eigen3::eigen_tensor_general<T>>>
#endif
    : Eigen::internal::traits<std::decay_t<T>> {};

} // OpenKalman::Eigen3::internal

#endif