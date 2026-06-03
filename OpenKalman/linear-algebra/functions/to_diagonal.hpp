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
#include "linear-algebra/adapters/to_diagonal_adapter.hpp"

namespace OpenKalman
{
  namespace interface
  {
    template<typename T>
    struct to_diagonal { static const bool is_specialized = false; };
  }



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
    if constexpr (interface::to_diagonal<stdex::remove_cvref_t<Arg>>::is_specialized)
    {
      return interface::to_diagonal<stdex::remove_cvref_t<Arg>>{}(std::forward<Arg>(arg));
    }
    else if constexpr (one_dimensional<Arg> and
      patterns::collection_compares_with<decltype(patterns::to_extents(p)), stdex::extents<std::size_t, 1>>)
    {
      return attach_patterns(std::forward<Arg>(arg), std::forward<P>(p));
    }
    else if constexpr (zero<Arg>)
    {
      return make_zero<values::value_type_of_t<element_type_of_t<Arg>>>(std::forward<P>(p));
    }
    else
    {
      return to_diagonal_adapter {std::forward<Arg>(arg), std::forward<P>(p)};
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
    if constexpr (compares_with_pattern_collection<Arg, patterns::Dimensions<1>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return to_diagonal(
        std::forward<Arg>(arg),
        patterns::views::to_diagonal(get_pattern_collection(arg)));
    }
  }


  namespace interface
  {
    template<typename Nested, typename PatternCollection>
    struct to_diagonal<pattern_adapter<Nested, PatternCollection>>
    {
      using NestedInterface = to_diagonal<stdex::remove_cvref_t<Nested>>;
      static const bool is_specialized = NestedInterface::is_specialized;

      template<typename Arg>
      constexpr auto
      operator()(Arg&& arg)
      {
        return NestedInterface{}(std::forward<Arg>(arg).nested_object(),
          patterns::views::to_diagonal(arg.pattern_collection()));
      }

    };

  }


}

#endif
