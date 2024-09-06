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
 * \brief Native Eigen3 traits for Eigen3 general FixedSizeAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_FIXEDSIZEADAPTER_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedObject, typename...Vs>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedObject, Vs...>>
#else
  template<typename NestedObject, typename...Vs>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedObject, Vs...>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedObject>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedObject>> {};

} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_FIXEDSIZEADAPTER_HPP
