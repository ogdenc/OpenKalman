/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref patterns::views::diagonal_of.
 */

#ifndef OPENKALMAN_PATTERNS_DIAGONAL_OF_HPP
#define OPENKALMAN_PATTERNS_DIAGONAL_OF_HPP

#include "collections/collections.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Any.hpp"
#include "patterns/concepts/compares_with.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/functions/get_pattern.hpp"
#include "patterns/traits/pattern_collection_element.hpp"

namespace OpenKalman::patterns::views
{
  namespace detail
  {
    /**
     *\internal
     * \brief A closure for the diagonal_of view.
     */
    struct diagonal_of_closure : stdex::ranges::range_adaptor_closure<diagonal_of_closure>
    {
      constexpr diagonal_of_closure() = default;

  #ifdef __cpp_concepts
    template<pattern_collection R> requires
      compares_with<
        pattern_collection_element_t<0, R>,
        pattern_collection_element_t<1, R>,
        &stdex::is_lteq, applicability::permitted> or
      compares_with<
        pattern_collection_element_t<1, R>,
        pattern_collection_element_t<0, R>,
        &stdex::is_lteq, applicability::permitted>
    constexpr pattern_collection decltype(auto)
  #else
    template<typename R, std::enable_if_t<
      pattern_collection<R> and
      (compares_with<
        pattern_collection_element_t<0, R>,
        pattern_collection_element_t<1, R>,
        &stdex::is_lteq, applicability::permitted> or
      compares_with<
        pattern_collection_element_t<1, R>,
        pattern_collection_element_t<0, R>,
        &stdex::is_lteq, applicability::permitted>), int> = 0>
    constexpr decltype(auto)
  #endif
      operator() (R&& r) const
      {
        using N2 = std::integral_constant<std::size_t, 2>;
        using P0 = pattern_collection_element_t<0, R>;
        using P1 = pattern_collection_element_t<1, R>;

        if constexpr (values::fixed_value_compares_with<collections::size_of<R>, 0>)
        {
          return std::forward<R>(r);
        }
        else if constexpr (values::fixed_value_compares_with<collections::size_of<R>, 1>)
        {
          return operator()(collections::views::concat(std::forward<R>(r), std::array{Dimensions<1>{}}));
        }
        else if constexpr (compares_with<P0, P1, &stdex::is_lteq>)
        {
          return collections::views::concat(
            std::array {get_pattern<0>(r)},
            collections::views::slice(std::forward<R>(r), N2{}));
        }
        else if constexpr (compares_with<P0, P1, &stdex::is_gt>)
        {
          return collections::views::concat(
            std::array {get_pattern<1>(r)},
            collections::views::slice(std::forward<R>(r), N2{}));
        }
        else
        {
          auto p0 = get_pattern<0>(r);
          auto p1 = get_pattern<1>(r);
          auto ps = collections::views::slice(std::forward<R>(r), N2{});

          if (compare<&stdex::is_lteq>(p0, p1))
            return collections::views::concat(std::array{Any{std::move(p0)}}, std::move(ps));
          if (compare<&stdex::is_gt>(p0, p1))
            return collections::views::concat(std::array{Any{std::move(p1)}}, std::move(ps));
          throw (std::logic_error("Patterns for the first two ranks are not compatible for taking diagonal"));
        }
      }
    };


    struct diagonal_of_adapter
    {
      constexpr auto
      operator() () const
      {
        return diagonal_of_closure {};
      }


  #ifdef __cpp_concepts
    template<pattern_collection R> requires
      compares_with<
        pattern_collection_element_t<0, R>,
        pattern_collection_element_t<1, R>,
        &stdex::is_lteq, applicability::permitted> or
      compares_with<
        pattern_collection_element_t<1, R>,
        pattern_collection_element_t<0, R>,
        &stdex::is_lteq, applicability::permitted>
    constexpr pattern_collection decltype(auto)
  #else
    template<typename R, std::enable_if_t<
      pattern_collection<R> and
      (compares_with<
        pattern_collection_element_t<0, R>,
        pattern_collection_element_t<1, R>,
        &stdex::is_lteq, applicability::permitted> or
      compares_with<
        pattern_collection_element_t<1, R>,
        pattern_collection_element_t<0, R>,
        &stdex::is_lteq, applicability::permitted>), int> = 0>
    constexpr decltype(auto)
  #endif
      operator() (R&& r) const
      {
        return diagonal_of_closure{}(std::forward<R>(r));
      }
    };

  }


  /**
   * \brief A RangeAdapterObject that converts one \ref pattern_collection to another corresponding to the \ref diagonal_matrix of the argument.
   * \details In the result, the pattern for rank 0 will be the the pattern for rank 0 in the argument,
   * except that it is potentially truncated if the argument's pattern for rank 1 is shorter.
   */
  inline constexpr detail::diagonal_of_adapter diagonal_of;

}

#endif
