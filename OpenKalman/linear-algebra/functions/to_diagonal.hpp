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
#include "linear-algebra/functions/attach_pattern.hpp"
#include "linear-algebra/functions/make_zero.hpp"
#include "linear-algebra/adapters/internal/owning_array.hpp"

namespace OpenKalman
{
  /**
   * \brief Convert a column vector (or any other array with a 1D second index) into a \ref diagonal_matrix.
   * \returns An \ref indexible_object with one higher rank than the argument
   */
#ifdef __cpp_concepts
  template<indexible Arg, patterns::pattern_collection P> requires
    pattern_collection_for<decltype(patterns::pattern_collection_of_diagonal(patterns::to_extents(std::declval<P>()))), Arg>
  constexpr diagonal_matrix decltype(auto)
#else
  template<typename Arg, typename P, std::enable_if_t<
    pattern_collection_for<decltype(patterns::pattern_collection_of_diagonal(patterns::to_extents(std::declval<P>()))), Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  to_diagonal(Arg&& arg, P&& p)
  {
    using extents_type = decltype(patterns::to_extents(p));
    if constexpr (one_dimensional<Arg> and
      patterns::collection_compares_with<extents_type, stdex::extents<std::size_t, 1>>)
    {
      return attach_pattern(std::forward<Arg>(arg), std::forward<P>(p));
    }
    else if constexpr (zero<Arg>)
    {
      return make_zero<values::value_type_of_t<element_type_of_t<Arg>>>(std::forward<P>(p));
    }
    else
    {
      decltype(auto) n = get_mdspan(arg);
      using N = std::decay_t<decltype(n)>;
      using nested_layout = typename N::layout_type;
      using nested_accessor = typename N::accessor_type;
      using mapping_type = typename interface::layout_to_diagonal<nested_layout>::template mapping<extents_type>;
      using accessor_type = interface::to_diagonal_accessor<nested_accessor>;
      auto nested_m = n.mapping();
      auto a = accessor_type {n.accessor(), nested_m.required_span_size()};
      auto m = mapping_type {std::move(nested_m), patterns::to_extents(p)};
      if constexpr (stdex::same_as<N, std::decay_t<Arg>>)
      {
        return stdex::mdspan(n.data_handle(), std::move(m), std::move(a));
      }
      else if constexpr (std::is_lvalue_reference_v<Arg>)
      {
        auto ret = attach_pattern(
          stdex::mdspan(n.data_handle(), std::move(m), std::move(a)),
          std::forward<P>(p));
        return ret;
      }
      else
      {
        auto ret = attach_pattern(
          internal::owning_array {std::forward<Arg>(arg), std::move(m), std::move(a)},
          std::forward<P>(p));
        return ret;
      }
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
  template<typename Arg, typename P, std::enable_if_t<indexible<Arg>, int> = 0>
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
        patterns::to_diagonal_pattern_collection(get_pattern_collection(arg)));
    }
  }


}

#endif
