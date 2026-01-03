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
 * \brief Definition for \ref hermitian_matrix.
 */

#ifndef OPENKALMAN_HERMITIAN_MATRIX_HPP
#define OPENKALMAN_HERMITIAN_MATRIX_HPP

#include "values/values.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/constant_object.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_explicitly_hermitian : std::false_type {};

    template<typename T>
    struct is_explicitly_hermitian<T, std::enable_if_t<interface::object_traits<stdex::remove_cvref_t<T>>::is_hermitian>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_inferred_hermitian_matrix : std::false_type {};

    template<typename T>
    struct is_inferred_hermitian_matrix<T, std::enable_if_t<values::not_complex<typename element_type_of<T>::type>>>
      : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix.
   * \details For rank >2 tensors, this must be applicable on every rank-2 slice comprising dimensions 0 and 1.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept hermitian_matrix =
    square_shaped<T, 2> and
    (interface::object_traits<stdex::remove_cvref_t<T>>::is_hermitian or
      ((constant_object<T> or diagonal_matrix<T>) and values::not_complex<element_type_of_t<T>>));
#else
  constexpr bool hermitian_matrix =
    square_shaped<T, 2> and
    (detail::is_explicitly_hermitian<T>::value or
      ((constant_object<T> or diagonal_matrix<T>) and detail::is_inferred_hermitian_matrix<T>::value));
#endif


}

#endif
