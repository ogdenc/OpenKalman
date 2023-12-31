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
 * \brief Eigen evaluators for \ref SelfContainedWrapper
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_SELFCONTAINEDWRAPPER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_SELFCONTAINEDWRAPPER_HPP


namespace Eigen::internal
{
  template<typename T, typename...Ps>
  struct evaluator<OpenKalman::internal::SelfContainedWrapper<T, Ps...>> : evaluator<std::decay_t<T>>
  {
    using XprType = OpenKalman::internal::SelfContainedWrapper<T, Ps...>;
    explicit evaluator(const XprType& t) : evaluator<std::decay_t<T>> {t.nested_object()} {}
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGEN_NATIVE_EVALUATORS_SELFCONTAINEDWRAPPER_HPP