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
 * \brief Definition for \ref constant_value.
 */

#ifndef OPENKALMAN_CONSTANT_VALUE_HPP
#define OPENKALMAN_CONSTANT_VALUE_HPP

#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "linear-algebra/traits/internal/get_singular_component.hpp"

namespace OpenKalman
{
  /**
   * \brief The constant value associated with a \ref constant_object or \ref constant_diagonal_object.
   */
#ifdef __cpp_concepts
  template<typename T> requires constant_object<T> or constant_diagonal_object<T>
  constexpr std::convertible_to<element_type_of_t<T>> auto
#else
  template<typename T, std::enable_if_t<constant_object<T> or constant_diagonal_object<T>, int> = 0>
  constexpr auto
#endif
  constant_value(T&& t)
  {
    using Trait = interface::object_traits<stdex::remove_cvref_t<T>>;
    if constexpr (interface::get_constant_defined_for<T>)
      return Trait::get_constant(std::forward<T>(t));
    else if constexpr (values::fixed<element_type_of_t<T>>)
      return element_type_of_t<T>{};
    else
      return internal::get_singular_component(std::forward<T>(t));
  }

}

#endif
