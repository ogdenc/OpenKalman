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
 * \brief Definition for \ref diagonal_of function.
 */

#ifndef OPENKALMAN_DIAGONAL_OF_HPP
#define OPENKALMAN_DIAGONAL_OF_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/concepts/compares_with_pattern_collection.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/functions/make_constant.hpp"
#include "linear-algebra/functions/internal/make_wrapped_mdspan.hpp"
#include "linear-algebra/interfaces/stl/diagonal_of_mdspan_policies.hpp"

namespace OpenKalman
{
  /**
   * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
   * \tparam Arg An \ref indexible object, which can have any rank and may or may not be square
   * \returns A column vector or slice corresponding to the diagonal.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    if constexpr (compares_with_pattern_collection<Arg, patterns::Dimensions<1>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::diagonal_of_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
    else
    {
      auto p = patterns::views::diagonal_of(get_pattern_collection(arg));
      if constexpr (constant_object<Arg> or constant_diagonal_object<Arg>)
      {
        return make_constant(constant_value(arg), std::move(p));
      }
      else
      {
        decltype(auto) n = get_mdspan(arg);
        using N = std::decay_t<decltype(n)>;
        using nested_extents_type = typename N::extents_type;
        using nested_layout = typename N::layout_type;

        using layout_type = interface::layout_diagonal_of<nested_layout, nested_extents_type>;
        using extents_type = decltype(patterns::to_extents(p));
        using mapping_type = typename layout_type::template mapping<extents_type>;
        auto map = mapping_type {n.mapping(), patterns::to_extents(p)};

        return internal::make_wrapped_mdspan(
          std::forward<Arg>(arg),
          stdex::identity{},
          std::move(map),
          n.accessor(),
          std::move(p));
      }
    }
  }


}

#endif
