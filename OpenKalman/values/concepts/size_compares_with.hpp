/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \values::size_compares_with.
 */

#ifndef OPENKALMAN_VALUES_SIZE_COMPARES_WITH_HPP
#define OPENKALMAN_VALUES_SIZE_COMPARES_WITH_HPP

#include "basics/basics.hpp"
#include "values/constants.hpp"
#include "fixed.hpp"
#include "size.hpp"
#include "values/traits/fixed_value_of.hpp"
#include "fixed_value_compares_with.hpp"

namespace OpenKalman::values
{
  namespace detail
  {
    template<auto comp, bool op_is_and = false, typename...Ords>
    constexpr bool
    do_comps(Ords...ords)
    {
      if constexpr (op_is_and) return (... and stdex::invoke(comp, ords));
      else return (... or stdex::invoke(comp, ords));
    }


    template<typename T, typename U, auto comp, applicability a>
    constexpr bool
    size_compares_with_impl()
    {
      constexpr bool unbt = not index<T>;
      constexpr bool unbu = not index<U>;
      constexpr bool ft = fixed_value_compares_with<T, stdex::dynamic_extent, &stdex::is_neq>;
      constexpr bool fu = fixed_value_compares_with<U, stdex::dynamic_extent, &stdex::is_neq>;

      if constexpr (unbt and unbu)
      {
        return do_comps<comp>(stdex::partial_ordering::equivalent);
      }
      else if constexpr (unbu)
      {
        return do_comps<comp>(stdex::partial_ordering::less);
      }
      else if constexpr (unbt)
      {
        return do_comps<comp>(stdex::partial_ordering::greater);
      }
      else if constexpr (ft and fu)
      {
#ifdef __cpp_impl_three_way_comparison
        return stdex::invoke(comp, fixed_value_of_v<T> <=> fixed_value_of_v<U>);
#else
        return stdex::invoke(comp, stdex::compare_three_way{}(fixed_value_of_v<T>, fixed_value_of_v<U>));
#endif
      }
      else if constexpr (ft)
      {
        if constexpr (fixed_value_compares_with<T, 0>)
          return do_comps<comp, a == applicability::guaranteed>(stdex::partial_ordering::less, stdex::partial_ordering::equivalent);
        else
          return a == applicability::permitted;
      }
      else if constexpr (fu)
      {
        if constexpr (fixed_value_compares_with<U, 0>)
          return do_comps<comp, a == applicability::guaranteed>(stdex::partial_ordering::greater, stdex::partial_ordering::equivalent);
        else
          return a == applicability::permitted;
      }
      else
      {
        return a == applicability::permitted;
      }
    }
  }


  /**
   * \brief T and U are sizes that compare in a particular way based on parameter comp.
   * \tparam comp A consteval-callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, typename U, auto comp = &stdex::is_eq, applicability a = applicability::guaranteed>
#ifdef __cpp_concepts
  concept size_compares_with =
#else
  constexpr bool size_compares_with =
#endif
    size<T> and
    size<U> and
    detail::size_compares_with_impl<T, U, comp, a>();

}

#endif
