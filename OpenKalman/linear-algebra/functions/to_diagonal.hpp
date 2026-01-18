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
 * \brief Definition for \ref to_diagonal function.
 */

#ifndef OPENKALMAN_TO_DIAGONAL_HPP
#define OPENKALMAN_TO_DIAGONAL_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/concepts/compares_with_pattern_collection.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/functions/make_zero.hpp"
#include "linear-algebra/functions/internal/make_wrapped_mdspan.hpp"
#include "linear-algebra/interfaces/stl/to_diagonal_mdspan_policies.hpp"

namespace OpenKalman
{
  /**
   * \brief Convert a column vector (or any other array with a 1D second index) into a \ref diagonal_matrix.
   * \tparam P A \ref patterns::pattern_collection equivalent to the intended resulting diagonal
   * \returns An \ref diagonal_matrix with a pattern equivalent equal to p.
   */
#ifdef __cpp_concepts
  template<indexible Arg, patterns::pattern_collection P> requires
    pattern_collection_for<decltype(patterns::views::diagonal_of(patterns::to_extents(std::declval<P>()))), Arg>
  constexpr diagonal_matrix decltype(auto)
#else
  template<typename Arg, typename P, std::enable_if_t<
    pattern_collection_for<decltype(patterns::views::diagonal_of(patterns::to_extents(std::declval<P>()))), Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  to_diagonal(Arg&& arg, P&& p)
  {
    using extents_type = decltype(patterns::to_extents(p));
    if constexpr (one_dimensional<Arg> and
      patterns::collection_compares_with<extents_type, stdex::extents<std::size_t, 1>>)
    {
      return attach_patterns(std::forward<Arg>(arg), std::forward<P>(p));
    }
    else if constexpr (zero<Arg>)
    {
      return make_zero<values::value_type_of_t<element_type_of_t<Arg>>>(std::forward<P>(p));
    }
    else
    {
      decltype(auto) n = get_mdspan(arg);
      using N = std::decay_t<decltype(n)>;
      using nested_extents_type = typename N::extents_type;
      using nested_layout = typename N::layout_type;
      using nested_accessor = typename N::accessor_type;
      auto nested_m = n.mapping();

      using layout_type = interface::layout_to_diagonal<nested_layout, nested_extents_type>;
      using mapping_type = typename layout_type::template mapping<extents_type>;
      std::size_t off_diag_index = nested_m.required_span_size();
      auto m = mapping_type {nested_m, patterns::to_extents(p)};

      using accessor_type = interface::to_diagonal_accessor<nested_accessor>;
      using data_handle_type = typename accessor_type::data_handle_type;
      auto a = accessor_type {n.accessor()};

      return internal::make_wrapped_mdspan(
        std::forward<Arg>(arg),
        [off_diag_index](auto&& h) { return data_handle_type {std::forward<decltype(h)>(h), off_diag_index}; },
        std::move(m),
        std::move(a),
        std::forward<P>(p));
    }
  }


  /**
   * \overload
   * \brief Simply duplicate the pattern for the first index.
   * \details The resulting matrix will be square in the first two ranks.
   * \returns An \ref indexible_object with one higher rank than the argument
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr diagonal_matrix decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  to_diagonal(Arg&& arg)
  {
    if constexpr (compares_with_pattern_collection<Arg, Dimensions<1>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::to_diagonal_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
    else
    {
      return to_diagonal(
        std::forward<Arg>(arg),
        patterns::views::to_diagonal(get_pattern_collection(arg)));
    }
  }


}

#endif
