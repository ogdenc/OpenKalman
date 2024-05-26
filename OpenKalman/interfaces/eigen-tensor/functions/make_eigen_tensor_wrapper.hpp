/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref make_eigen_wrapper function.
 */

#ifndef OPENKALMAN_MAKE_EIGEN_TENSOR_WRAPPER_HPP
#define OPENKALMAN_MAKE_EIGEN_TENSOR_WRAPPER_HPP

namespace OpenKalman::Eigen3
{
  /**
   * Make a \ref LibraryWrapper for the Eigen tensor library.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  inline auto
  make_eigen_tensor_wrapper(Arg&& arg)
  {
    return EigenTensorWrapper<Arg> {std::forward<Arg>(arg)};
  }

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_MAKE_EIGEN_TENSOR_WRAPPER_HPP
