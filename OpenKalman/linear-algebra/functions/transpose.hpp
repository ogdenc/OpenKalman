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
 * \brief Definition of transpose function.
 */

#ifndef OPENKALMAN_TRANSPOSE_HPP
#define OPENKALMAN_TRANSPOSE_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/functions/make_constant.hpp"
#include "linear-algebra/functions/internal/make_wrapped_mdspan.hpp"
#include "linear-algebra/functions/conjugate.hpp"
#include "linear-algebra/interfaces/stl/transpose_mdspan_policies.hpp"

namespace OpenKalman
{
  /**
   * \brief Swap any two indices of an \ref indexible_object
   * \details By default, the first two indices are transposed.
   * \tparam indexa The first index to be be transposed.
   * \tparam indexb The second index to be be transposed.
   */
#ifdef __cpp_concepts
  template<std::size_t indexa = 0, std::size_t indexb = 1, indexible Arg> requires (indexa < indexb)
  constexpr indexible decltype(auto)
#else
  template<std::size_t indexa = 0, std::size_t indexb = 1, typename Arg, std::enable_if_t<
    indexible<Arg> and (indexa < indexb), int> = 0>
  constexpr decltype(auto)
#endif
  transpose(Arg&& arg)
  {
    using P = decltype(get_pattern_collection(arg));
    if constexpr (
      patterns::compares_with<patterns::pattern_collection_element_t<indexa, P>, patterns::pattern_collection_element_t<indexb, P>> and
      (constant_object<Arg> or
        (indexb == 1 and
          (diagonal_matrix<Arg> or
            (hermitian_matrix<Arg> and not values::complex<element_type_of_t<Arg>>)))))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::transpose_defined_for<Arg&&, indexa, indexb>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::template transpose<indexa, indexb>(std::forward<Arg>(arg));
    }
    else if constexpr (constant_object<Arg>)
    {
      return make_constant(constant_value(arg), patterns::views::transpose<indexa, indexb>(get_pattern_collection(arg)));
    }
    else if constexpr (hermitian_matrix<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else
    {
      auto p = patterns::views::transpose<indexa, indexb>(get_pattern_collection(arg));
      auto n = get_mdspan(arg);
      using nested_layout = typename std::decay_t<decltype(n)>::layout_type;
      using extents_type = std::decay_t<decltype(to_extents(p))>;
      if constexpr (indexb == 1 and n.rank() == 2)
      {
        using layout_type = stdex::linalg::layout_transpose<nested_layout>;
        using mapping_type = typename layout_type::template mapping<extents_type>;
        return internal::make_wrapped_mdspan(
          std::forward<Arg>(arg),
          stdex::identity{},
          mapping_type(n.mapping()),
          n.accessor(),
          std::move(p));
      }
      else
      {
        using nested_mapping_type = typename std::decay_t<decltype(n)>::mapping_type;
        using layout_type = interface::layout_transpose<nested_mapping_type, indexa, indexb>;
        using mapping_type = typename layout_type::template mapping<extents_type>;
        return internal::make_wrapped_mdspan(
          std::forward<Arg>(arg),
          stdex::identity{},
          mapping_type(n.mapping(), to_extents(p)),
          n.accessor(),
          std::move(p));
      }
    }
  }


}

#endif
