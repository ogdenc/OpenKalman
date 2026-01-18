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
 * \brief Definition of \ref compare_indices
 */

#ifndef OPENKALMAN_COLLECTION_COMPARE_INDICES_HPP
#define OPENKALMAN_COLLECTION_COMPARE_INDICES_HPP

#include "values/values.hpp"
#include "collections/concepts/index.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename Lhs, typename Rhs>
    constexpr auto
    get_min_size(const Lhs& lhs, const Rhs& rhs)
    {
      struct Op
      {
        constexpr auto
        operator()(std::size_t a, std::size_t b) const { return std::min(a, b); }
      };

      if constexpr (sized<Lhs> and sized<Rhs>)
        return values::operation(Op{}, get_size(lhs), get_size(rhs));
      else if constexpr (sized<Lhs>)
        return get_size(lhs);
      else
        return get_size(rhs);
    }


    template<typename Lhs, typename Rhs>
    constexpr std::size_t
    get_fixed_size()
    {
      constexpr bool fixed_lhs = values::fixed_value_compares_with<size_of<Lhs>, stdex::dynamic_extent, &stdex::is_neq>;
      constexpr bool fixed_rhs = values::fixed_value_compares_with<size_of<Rhs>, stdex::dynamic_extent, &stdex::is_neq>;
      if constexpr (fixed_lhs and fixed_rhs)
        return std::min(size_of_v<Lhs>, size_of_v<Rhs>);
      else if constexpr (fixed_lhs)
        return size_of_v<Lhs>;
      else if constexpr (fixed_rhs)
        return size_of_v<Rhs>;
      else
        return stdex::dynamic_extent;
    }


    template<auto comp>
    struct compare_indices_fixed_op
    {
      template<typename A, typename B>
      constexpr auto operator()(const A& a, const B& b) const
      {
        return stdex::invoke(comp, stdex::compare_three_way{}(a, b));
      }
    };


    template<auto comp, std::size_t fsz, std::size_t i = 0, typename Lhs, typename Rhs, typename Sz>
    constexpr auto
    compare_indices_fixed(const Lhs& lhs, const Rhs& rhs, const Sz& sz)
    {
      using ix = std::integral_constant<std::size_t, i>;

      if constexpr (values::size_compares_with<ix, Sz, &stdex::is_lt>)
        return values::operation(
          std::logical_and{},
          values::operation(compare_indices_fixed_op<comp>{}, collections::get<i>(lhs), collections::get<i>(rhs)),
          compare_indices_fixed<comp, fsz, i + 1>(lhs, rhs, sz));
      else if constexpr (values::size_compares_with<ix, Sz, &stdex::is_gteq> or i >= fsz)
        return std::true_type {};
      else
      {
        if (i < sz)
          return compare_indices_fixed_op<comp>{}(collections::get<i>(lhs), collections::get<i>(rhs)) and
            compare_indices_fixed<comp, fsz, i + 1>(lhs, rhs, sz);
        else
          return true;
      }
    }

  }


  /**
   * \brief Performs an element-by-element comparison of two \ref collections::index "index collections".
   * \details If one argument is longer than the other, the comparison will ignore the tail of the longer argument.
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
#ifdef __cpp_concepts
  template<auto comp = &stdex::is_eq, index Lhs, index Rhs> requires sized<Lhs> or sized<Rhs>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<auto comp = &stdex::is_eq, typename Lhs, typename Rhs, std::enable_if_t<
    index<Lhs> and index<Rhs> and (sized<Lhs> or sized<Rhs>), int> = 0>
  constexpr auto
#endif
  compare_indices(const Lhs& lhs, const Rhs& rhs)
  {
    auto sz = detail::get_min_size(lhs, rhs);
    constexpr std::size_t fsz = detail::get_fixed_size<Lhs, Rhs>();

    if constexpr (fsz != stdex::dynamic_extent)
    {
      return detail::compare_indices_fixed<comp, fsz>(lhs, rhs, sz);
    }
    else
    {
      for (std::size_t i = 0; i < sz; ++i)
      {
        if (not stdex::invoke(comp, stdex::compare_three_way{}(collections::get_element(lhs, i), collections::get_element(rhs, i))))
          return false;
      }
      return true;
    }
  }


}

#endif
