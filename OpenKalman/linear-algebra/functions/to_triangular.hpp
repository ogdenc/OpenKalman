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
 * \brief Definitions for \ref to_triangular.
 */

#ifndef OPENKALMAN_TO_TRIANGULAR_HPP
#define OPENKALMAN_TO_TRIANGULAR_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/adapters/triangular_adapter.hpp"

namespace OpenKalman
{
  namespace interface
  {
    template<typename T, triangle_type tri>
    struct to_triangular { static const bool is_specialized = false; };
  }


  /**
   * \brief Convert an object to a \ref triangular_matrix.
   * \details Any element outside the defined triangle will be zero.
   * The argument may have more than two indices, in which case every slice
   * involving the first two indices will be triangular.
   * \tparam t The \ref triangle_type of the result.
   * \tparam Arg The argument.
   */
#ifdef __cpp_concepts
  template<triangle_type t, indexible Arg> requires (t != triangle_type::none)
  constexpr triangular_matrix<t> decltype(auto)
#else
  template<triangle_type t, typename Arg, std::enable_if_t<indexible<Arg> and (t != triangle_type::none), int> = 0>
  constexpr decltype(auto)
#endif
  to_triangular(Arg&& arg)
  {
    if constexpr (triangular_matrix<Arg, t>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::to_triangular<stdex::remove_cvref_t<Arg>, t>::is_specialized)
    {
      return interface::to_triangular<stdex::remove_cvref_t<Arg>, t>{}(std::forward<Arg>(arg));
    }
    else
    {
      return triangular_adapter<Arg, t>(std::forward<Arg>(arg));
    }
  }


  namespace interface
  {
    template<typename Nested, typename PatternCollection, triangle_type tri>
    struct to_triangular<pattern_adapter<Nested, PatternCollection>, tri>
    {
      using NestedInterface = to_triangular<stdex::remove_cvref_t<Nested>, tri>;

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
