/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of conjugate_transpose function.
 */

#ifndef OPENKALMAN_CONJUGATE_TRANSPOSE_HPP
#define OPENKALMAN_CONJUGATE_TRANSPOSE_HPP

#include "patterns/patterns.hpp"
#include "conjugate.hpp"
#include "transpose.hpp"

namespace OpenKalman
{
  /**
   * \brief Take the conjugate-transpose of an \ref indexible_object
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  conjugate_transpose(Arg&& arg)
  {
    using P = decltype(get_pattern_collection(arg));
    constexpr bool square = patterns::compares_with<patterns::pattern_collection_element_t<0, P>, patterns::pattern_collection_element_t<1, P>>;
    if constexpr (hermitian_matrix<Arg> and square)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr ((diagonal_matrix<Arg> or constant_object<Arg>) and square)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (not values::complex<element_type_of_t<Arg>> or values::not_complex<constant_value_of<Arg>>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (interface::conjugate_transpose_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::conjugate_transpose(std::forward<Arg>(arg));
    }
    else
    {
      return conjugate(transpose(std::forward<Arg>(arg)));
    }
  }

}

#endif
