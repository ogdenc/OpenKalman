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
 * \brief Definition for \ref identity_matrix.
 */

#ifndef OPENKALMAN_IDENTITY_MATRIX_HPP
#define OPENKALMAN_IDENTITY_MATRIX_HPP

#include "values/values.hpp"
#include "constant_diagonal_object.hpp"
#include "linear-algebra/traits/constant_value.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_identity_matrix : std::false_type {};

    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<
      constant_diagonal_object<T> and
      values::fixed_value_compares_with<decltype(constant_value(std::declval<T&>())), 1>>>
      : std::true_type {};
  }
#endif

  /**
   * \brief Specifies that a type is known at compile time to be a rank-2 or lower identity matrix.
   * \details This is a generalized identity matrix which may be rectangular (with zeros in all non-diagonal components.
   * A 1D vector with constant element 1 and an \ref empty_matrix are also considered to be identity matrices.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix =
    constant_diagonal_object<T> and
    values::fixed_value_compares_with<decltype(constant_value(std::declval<T&>())), 1>;
#else
    constexpr bool identity_matrix =
      detail::is_identity_matrix<T>::value;
#endif


}

#endif
