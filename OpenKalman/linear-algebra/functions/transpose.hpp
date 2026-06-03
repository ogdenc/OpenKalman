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
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/functions/make_constant.hpp"
#include "linear-algebra/functions/conjugate.hpp"
#include "linear-algebra/adapters/transpose_adapter.hpp"

namespace OpenKalman
{
  namespace interface
  {
    template<typename T, std::size_t indexa, std::size_t indexb>
    struct transpose { static const bool is_specialized = false; };
  }


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
    constexpr bool square = patterns::compares_with<patterns::pattern_collection_element_t<indexa, P>, patterns::pattern_collection_element_t<indexb, P>>;
    if constexpr (square and
      (constant_object<Arg> or
        (indexb == 1 and
          (diagonal_matrix<Arg> or
            (hermitian_matrix<Arg> and not values::complex<element_type_of_t<Arg>>)))))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::transpose<stdex::remove_cvref_t<Arg>, indexa, indexb>::is_specialized)
    {
      return interface::transpose<stdex::remove_cvref_t<Arg>, indexa, indexb>{}(std::forward<Arg>(arg));
    }
    else if constexpr (constant_object<Arg>)
    {
      return make_constant(constant_value(arg), patterns::views::transpose<indexa, indexb>(get_pattern_collection(arg)));
    }
    else if constexpr (constant_diagonal_object<Arg>)
    {
      return make_constant_diagonal(constant_value(arg), patterns::views::transpose<indexa, indexb>(get_pattern_collection(arg)));
    }
    else if constexpr (hermitian_matrix<Arg> and square)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return to_diagonal(diagonal_of(std::forward<Arg>(arg)), patterns::views::transpose<indexa, indexb>(get_pattern_collection(arg)));
    }
    else
    {
      return transpose_adapter<Arg, indexa, indexb>{std::forward<Arg>(arg)};
    }
  }


  namespace interface
  {
    template<typename Nested, typename PatternCollection, std::size_t indexa, std::size_t indexb>
    struct transpose<pattern_adapter<Nested, PatternCollection>, indexa, indexb>
    {
      using NestedInterface = transpose<stdex::remove_cvref_t<Nested>, indexa, indexb>;

      static const bool is_specialized = NestedInterface::is_specialized;

      template<typename Arg>
      constexpr auto
      operator()(Arg&& arg)
      {
        return attach_patterns(
          NestedInterface{}(std::forward<Arg>(arg).nested_object()),
          patterns::views::transpose<indexa, indexb>(arg.pattern_collection()) );
      }

    };

  }


}

#endif
