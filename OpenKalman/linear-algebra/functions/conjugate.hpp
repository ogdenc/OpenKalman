/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of conjugate function.
 */

#ifndef OPENKALMAN_CONJUGATE_HPP
#define OPENKALMAN_CONJUGATE_HPP

#include "values/values.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/zero.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "linear-algebra/traits/constant_value_of.hpp"
#include "linear-algebra/concepts/identity_matrix.hpp"
#include "make_constant.hpp"
//#include "to_diagonal.hpp"

namespace OpenKalman
{
  /**
   * \brief Take the complex conjugate of an \ref indexible object
   * \details The resulting object has every element substituted with its complex conjugate.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  conjugate(Arg&& arg)
  {
    if constexpr (not values::complex<element_type_of_t<Arg>> or values::not_complex<constant_value_of<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::conjugate_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (std::is_lvalue_reference_v<Arg>)
    {
      return conjugate(get_mdspan(arg));
    }
    else if constexpr (constant_object<Arg>)
    {
      return make_constant(values::conj(constant_value(arg)), get_pattern_collection(std::forward<Arg>(arg)));
    }
    else if constexpr (constant_diagonal_object<Arg>)
    {
      return to_diagonal(make_constant(values::conj(constant_value(arg)), get_pattern_collection(std::forward<Arg>(arg))));
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      //return to_diagonal(conjugate(diagonal_of(std::forward<Arg>(arg))));
    }
    else
    {
      static_assert(interface::conjugate_defined_for<Arg&&>, "Interface not defined");
    }
  }


}

#endif
