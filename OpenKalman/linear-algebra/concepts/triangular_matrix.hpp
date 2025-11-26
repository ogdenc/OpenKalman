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
 * \brief Definition for \ref triangular_matrix.
 */

#ifndef OPENKALMAN_TRIANGULAR_MATRIX_HPP
#define OPENKALMAN_TRIANGULAR_MATRIX_HPP

#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/triangle_type_of.hpp"

namespace OpenKalman
{
  /**
   * \brief Specifies that an argument is an \ref indexible object having
   * a given \ref triangle_type (e.g., upper, lower, or diagonal).
   * \details This can also test whether the argument is triangular (\ref triangle_type::none).
   * To test whether the argument has any triangle type, use \ref triangle_type::any (which is the default).
   * Diagonal matrices are considered to also be both upper and lower.
   * \tparam t The \ref triangle_type
   */
  template<typename T, triangle_type t = triangle_type::any>
#ifdef __cpp_concepts
  concept triangular_matrix =
#else
  constexpr bool triangular_matrix =
#endif
    indexible<T> and
    (t == triangle_type_of_v<T> or
      (t == triangle_type::any and triangle_type_of_v<T> != triangle_type::none) or
      ((t == triangle_type::upper or t == triangle_type::lower) and triangle_type_of_v<T> == triangle_type::diagonal));


}

#endif
