/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref constant_diagonal_object.
 */

#ifndef OPENKALMAN_CONSTANT_DIAGONAL_OBJECT_HPP
#define OPENKALMAN_CONSTANT_DIAGONAL_OBJECT_HPP

#include "values/values.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/concepts/zero.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"

namespace OpenKalman
{
  /**
   * \brief Specifies that all diagonal elements of a diagonal object are known at compile time to be the same constant value.
   * \details A constant diagonal matrix is also a \ref diagonal_matrix. It is not necessarily square.
   * If T is a rank >2 tensor, every rank-2 slice comprising dimensions 0 and 1 must be constant diagonal matrix.
   * To check if the constant is zero, it is better to use \ref zero, which might catch more cases.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept constant_diagonal_object =
#else
  constexpr bool constant_diagonal_object =
#endif
    indexible<T> and
    ((interface::get_constant_defined_for<T> and diagonal_matrix<T>) or
      zero<T> or
      one_dimensional<T>);

}

#endif
