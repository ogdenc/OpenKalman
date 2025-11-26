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
 * \brief Definition for \ref constant_object.
 */

#ifndef OPENKALMAN_CONSTANT_OBJECT_HPP
#define OPENKALMAN_CONSTANT_OBJECT_HPP

#include "values/values.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"
#include "linear-algebra/concepts/zero.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct element_type_is_fixed : std::false_type {};

    template<typename T>
    struct element_type_is_fixed<T, std::enable_if_t<values::fixed<typename element_type_of<T>::type>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that all elements of an object are known at compile time to be the same constant value.
   * \details To check if the constant is zero, it is better to use \ref zero.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept constant_object =
    indexible<T> and
    ((interface::get_constant_defined_for<T> and not triangular_matrix<T>) or
      values::fixed<element_type_of_t<T>> or
      zero<T> or
      one_dimensional<T>);
#else
  constexpr bool constant_object =
    indexible<T> and
    ((interface::get_constant_defined_for<T> and not triangular_matrix<T>) or
      detail::element_type_is_fixed<T>::value or
      zero<T> or
      one_dimensional<T>);
#endif

}

#endif
