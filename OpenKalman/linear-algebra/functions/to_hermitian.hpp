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
 * \brief Definitions for \ref to_hermitian.
 */

#ifndef OPENKALMAN_TO_HERMITIAN_HPP
#define OPENKALMAN_TO_HERMITIAN_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/traits/constant_value_of.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
#include "linear-algebra/adapters/hermitian_adapter.hpp"

namespace OpenKalman
{
  namespace interface
  {
    template<typename T, triangle_type tri>
    struct to_hermitian { static const bool is_specialized = false; };
  }


  /**
   * \brief Creates a \ref hermitian_matrix by, if necessary, wrapping the argument in a \ref hermitian_adapter.
   * \note The result is guaranteed to be either hermitian or a \ref hermitian_adapter.
   * \tparam storage_type The intended \ref triangle_type of the result (lower or upper).
   * \tparam Arg A square matrix.
   */
#ifdef __cpp_concepts
  template<triangle_type storage_type, square_shaped<2, applicability::permitted> Arg> requires 
    (storage_type == triangle_type::lower or storage_type == triangle_type::upper) and 
    (not (constant_object<Arg> or constant_diagonal_object<Arg>) or
      not values::fixed<constant_value_of<Arg>> or
      values::not_complex<constant_value_of<Arg>>)
  constexpr hermitian_matrix decltype(auto)
#else
  template<triangle_type storage_type = triangle_type::lower, typename Arg, std::enable_if_t<
    square_shaped<Arg, 2, applicability::permitted> and
    (storage_type == triangle_type::lower or storage_type == triangle_type::upper) and 
    (not (constant_object<Arg> or constant_diagonal_object<Arg>) or
      not values::fixed<constant_value_of<Arg>> or
      values::not_complex<constant_value_of<Arg>>), int> = 0>
  constexpr decltype(auto)
#endif
  to_hermitian(Arg&& arg)
  {
    if constexpr (hermitian_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::to_hermitian<stdex::remove_cvref_t<Arg>, storage_type>::is_specialized)
    {
      return interface::to_hermitian<stdex::remove_cvref_t<Arg>, storage_type>{}(std::forward<Arg>(arg));
    }
    else
    {
      return hermitian_adapter<Arg, storage_type> {std::forward<Arg>(arg)};
    }
  }


  namespace interface
  {
    template<typename Nested, typename PatternCollection, triangle_type tri>
    struct to_hermitian<pattern_adapter<Nested, PatternCollection>, tri>
    {
      using NestedInterface = to_hermitian<stdex::remove_cvref_t<Nested>, tri>;

      static const bool is_specialized = NestedInterface::is_specialized;

      template<typename Arg>
      constexpr auto
      operator()(Arg&& arg)
      {
        return attach_patterns(
          NestedInterface{}(std::forward<Arg>(arg).nested_object()),
          arg.pattern_collection() );
      }

    };

  }


}

#endif
