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
 * \brief Definition for \ref constant_matrix.
 */

#ifndef OPENKALMAN_CONSTANT_MATRIX_HPP
#define OPENKALMAN_CONSTANT_MATRIX_HPP

#include "indexible.hpp"
#include "linear-algebra/traits/constant_coefficient.hpp"

namespace OpenKalman
{
  /**
   * \brief Specifies that all components of an object are the same constant value.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept constant_matrix =
#else
  constexpr bool constant_matrix =
#endif
    indexible<T> and value::scalar<constant_coefficient<T>>;


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_MATRIX_HPP
