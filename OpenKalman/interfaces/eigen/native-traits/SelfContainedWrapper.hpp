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
 * \brief Eigen traits relating to \ref SelfContainedWrapper
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_SELFCONTAINEDWRAPPER_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_SELFCONTAINEDWRAPPER_HPP

namespace Eigen::internal
{
  template<typename T, typename...Ps>
  struct traits<OpenKalman::internal::SelfContainedWrapper<T, Ps...>> : traits<std::decay_t<T>>
  {
    enum {
      Flags = (std::decay_t<T>::Flags & ~NestByRefBit),
    };
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_SELFCONTAINEDWRAPPER_HPP