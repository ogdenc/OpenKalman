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
#include "linear-algebra/concepts/compares_with_pattern_collection.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "linear-algebra/functions/make_constant.hpp"
#include "linear-algebra/adapters/diagonal_of_adapter.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"

namespace OpenKalman
{
  namespace interface
  {
    template<typename T>
    struct diagonal_of { static const bool is_specialized = false; };
  }


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
    else if constexpr (interface::diagonal_of<stdex::remove_cvref_t<Arg>>::is_specialized)
    {
      return interface::diagonal_of<stdex::remove_cvref_t<Arg>>{}(std::forward<Arg>(arg));
    }
    else if constexpr (constant_object<Arg> or constant_diagonal_object<Arg>)
    {
      return make_constant(
        constant_value(std::forward<Arg>(arg)),
        patterns::views::diagonal_of(get_pattern_collection(arg)));
    }
    else
    {
      return diagonal_of_adapter {std::forward<Arg>(arg)};
    }
  }


  namespace interface
  {
    template<typename Nested, typename PatternCollection>
    struct diagonal_of<pattern_adapter<Nested, PatternCollection>>
    {
      using NestedInterface = diagonal_of<stdex::remove_cvref_t<Nested>>;
      static const bool is_specialized = NestedInterface::is_specialized;

      template<typename Arg>
      constexpr auto
      operator()(Arg&& arg)
      {
        return attach_patterns(
          NestedInterface{}(std::forward<Arg>(arg).nested_object()),
          patterns::views::diagonal_of(arg.pattern_collection()) );
      }

    };

  }

}

#endif
