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
 * \brief Definition of transpose function.
 */

#ifndef OPENKALMAN_TRANSPOSE_HPP
#define OPENKALMAN_TRANSPOSE_HPP

#include "values/values.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "conjugate.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename C, typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_constant(C&& c, Arg&& arg, std::index_sequence<Is...>)
    {
      return make_constant<Arg>(std::forward<C>(c),
        get_pattern_collection<1>(arg), get_pattern_collection<0>(arg), get_pattern_collection<Is + 2>(arg)...);
    }
  }


  /**
   * \brief Swap any two indices of an \ref indexible_object
   * \details By default, the first two indices are transposed.
   */
#ifdef __cpp_concepts
  template<std::size_t indexa = 0, std::size_t indexb = 1, indexible Arg> requires (indexa < indexb)
#else
  template<std::size_t indexa = 0, std::size_t indexb = 1, typename Arg, std::enable_if_t<
    indexible<Arg> and (indexa < indexb), int> = 0>
#endif
  constexpr decltype(auto) transpose(Arg&& arg)
  {
    constexpr bool square_matrix = values::size_compares_with<index_dimension_of<Arg, 0>, index_dimension_of<Arg, 1>>;
    constexpr bool diag_invariant = (diagonal_matrix<Arg> or constant_object<Arg>) and square_matrix;
    constexpr bool hermitian_invariant = hermitian_matrix<Arg> and not values::complex<element_type_of_t<Arg>>;
    constexpr bool invariant = diag_invariant or hermitian_invariant;

    if constexpr (invariant)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (indexb == 1 and interface::matrix_transpose_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
    else if constexpr (interface::transpose_defined_for<Arg&&, indexa, indexb>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::template transpose<indexa, indexb>(std::forward<Arg>(arg));
    }
    else if constexpr (constant_object<Arg>)
    {
      constexpr std::make_index_sequence<std::max({index_count_v<Arg>, 2_uz}) - 2_uz> seq;
      return detail::transpose_constant(constant_value(arg), std::forward<Arg>(arg), seq);
    }
    else if constexpr (hermitian_matrix<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (std::is_lvalue_reference_v<Arg>)
    {
      return attach_pattern(transpose(get_mdspan(arg)), get_pattern_collection(std::forward<Arg>(arg)));
    }
    else if (indexb == 1)
    {
      static_assert(interface::matrix_transpose_defined_for<Arg&&>, "Interface not defined");
    }
    else
    {
      static_assert(interface::transpose_defined_for<Arg&&, indexa, indexb>, "Interface not defined");
    }
  }


}

#endif
