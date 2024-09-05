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
 * \brief Definition for \ref identity_matrix.
 */

#ifndef OPENKALMAN_IDENTITY_MATRIX_HPP
#define OPENKALMAN_IDENTITY_MATRIX_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_identity_matrix : std::false_type {};

    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<constant_diagonal_matrix<T, ConstantType::static_constant>>>
      : std::bool_constant<internal::are_within_tolerance(constant_diagonal_coefficient_v<T>, 1)> {};
  }
#endif

  /**
   * \brief Specifies that a type is an identity matrix.
   * \details This is a generalized identity matrix which may be rectangular (with zeros in all non-diagonal components.
   * For rank >2 tensors, every rank-2 slice comprising dimensions 0 and 1 must be an identity matrix as defined here.
   * Every \ref empty_object is also an identity matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix =
    (constant_diagonal_matrix<T, ConstantType::static_constant> and internal::are_within_tolerance(constant_diagonal_coefficient_v<T>, 1)) or
#else
    constexpr bool identity_matrix = detail::is_identity_matrix<T>::value or
#endif
    empty_object<T>;


} // namespace OpenKalman

#endif //OPENKALMAN_IDENTITY_MATRIX_HPP
