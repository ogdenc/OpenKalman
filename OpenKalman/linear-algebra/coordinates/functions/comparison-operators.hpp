/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Comparison operators for \rev coordinates::pattern objects.
 */

#ifndef OPENKALMAN_COORDINATES_COMPARISON_OPERATORS_HPP
#define OPENKALMAN_COORDINATES_COMPARISON_OPERATORS_HPP

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif
#include <iostream>
#include <type_traits>
#include "basics/global-definitions.hpp"
#include "values/functions/to_number.hpp"
#include "collections/views/single.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/internal/get_hash_code.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_collection_element.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
#ifdef __cpp_impl_three_way_comparison
    using ordering = std::partial_ordering;

    template<typename A, typename B>
    static constexpr std::partial_ordering
    ordering_compare(const A& a, const B& b)
    {
      return a <=> b;
    }
#else
    namespace cmp_cat
    {
      using type = signed char;
      enum struct Ord : type { equivalent = 0, less = -1, greater = 1 };
      enum struct Ncmp : type { unordered = 2 };
    }

    struct ordering
    {
      cmp_cat::type m_value;

      constexpr explicit
      ordering(cmp_cat::Ord v) noexcept : m_value(cmp_cat::type(v)) {}

      constexpr explicit
      ordering(cmp_cat::Ncmp v) noexcept : m_value(cmp_cat::type(v)) {}

    public:

      static const ordering less;
      static const ordering equivalent;
      static const ordering greater;
      static const ordering unordered;

      [[nodiscard]] friend constexpr bool
      operator==(ordering v, ordering w) noexcept { return v.m_value == w.m_value; }
    };

    constexpr ordering ordering::less(cmp_cat::Ord::less);
    constexpr ordering ordering::equivalent(cmp_cat::Ord::equivalent);
    constexpr ordering ordering::greater(cmp_cat::Ord::greater);
    constexpr ordering ordering::unordered(cmp_cat::Ncmp::unordered);


    template<typename A, typename B>
    static constexpr ordering
    ordering_compare(const A& a, const B& b)
    {
      return
        a < b ? ordering::less :
        a > b ? ordering::greater :
        a == b ? ordering::equivalent :
        ordering::unordered;
    }
#endif


    template<std::size_t ai = 0, std::size_t bi = 0, std::size_t abank = 0, std::size_t bbank = 0, typename A, typename B>
    constexpr auto
    compare_fixed(const A& a, const B& b)
    {
      if constexpr (ai < collections::size_of_v<A> and bi < collections::size_of_v<B>)
      {
        using Ai = std::tuple_element_t<ai, A>;
        using Bi = std::tuple_element_t<bi, B>;
        constexpr bool ae = euclidean_pattern<Ai>, be = euclidean_pattern<Bi>;
        if constexpr (ae and be) return compare_fixed<ai + 1, bi + 1, abank + dimension_of_v<Ai>, bbank + dimension_of_v<Bi>>(a, b);
        else if constexpr (ae) return compare_fixed<ai + 1, bi, abank + dimension_of_v<Ai>, bbank>(a, b);
        else if constexpr (be) return compare_fixed<ai, bi + 1, abank, bbank + dimension_of_v<Bi>>(a, b);
        else
        {
          if constexpr (abank == bbank) if (internal::get_hash_code(collections::get<ai>(a)) == internal::get_hash_code(collections::get<bi>(b)))
            return compare_fixed<ai + 1, bi + 1>(a, b);
          return ordering::unordered;
        }
      }
      else if constexpr (ai < collections::size_of_v<A>) // bi >= collections::size_of_v<B>
      {
        using Ai = std::tuple_element_t<ai, A>;
        if (euclidean_pattern<Ai>) return compare_fixed<ai + 1, bi, abank + dimension_of_v<Ai>, bbank>(a, b);
        if (values::to_number(abank) >= values::to_number(bbank)) return ordering::greater;
        return ordering::unordered;
      }
      else if constexpr (bi < collections::size_of_v<B>) // ai >= collections::size_of_v<A>
      {
        using Bi = std::tuple_element_t<bi, B>;
        if (euclidean_pattern<Bi>) return compare_fixed<ai, bi + 1, abank, bbank + dimension_of_v<Bi>>(a, b);
        if (values::to_number(abank) <= values::to_number(bbank)) return ordering::less;
        return ordering::unordered;
      }
      else
      {
        // not ai_in_range and not bi_in_range:
        return ordering_compare(values::to_number(abank), values::to_number(bbank));
      }
    }


    template<typename A, typename B>
    constexpr ordering
    compare_impl(const A& a, const B& b, std::size_t ai = 0, std::size_t bi = 0, std::size_t abank = 0, std::size_t bbank = 0)
    {
      bool ai_in_range = ai < values::to_number(get_size(a));
      bool bi_in_range = bi < values::to_number(get_size(b));

      if (ai_in_range and bi_in_range)
      {
        auto a_i = internal::get_descriptor_collection_element(a, ai);
        auto b_i = internal::get_descriptor_collection_element(b, bi);
        bool ae = get_is_euclidean(a_i), be = get_is_euclidean(b_i);
        if (ae or be) return compare_impl(a, b, (ae ? ai + 1 : ai), (be ? bi + 1 : bi),
          (ae ? abank + get_dimension(a_i) : abank), (be ? bbank + get_dimension(b_i) : bbank));
        if (internal::get_hash_code(a_i) == internal::get_hash_code(b_i) and abank == bbank) return compare_impl(a, b, ai + 1, bi + 1);
        return ordering::unordered;
      }
      if (ai_in_range) // not bi_in_range
      {
        auto a_i = internal::get_descriptor_collection_element(a, ai);
        if (get_is_euclidean(a_i)) return compare_impl(a, b, ai + 1, bi, abank + get_dimension(a_i), bbank);
        if (abank >= bbank) return ordering::greater;
        return ordering::unordered;
      }
      if (bi_in_range) // not ai_in_range
      {
        auto b_i = internal::get_descriptor_collection_element(b, bi);
        if (get_is_euclidean(b_i)) return compare_impl(a, b, ai, bi + 1, abank, bbank + get_dimension(b_i));
        if (abank <= bbank) return ordering::less;
        return ordering::unordered;
      }
      // not ai_in_range and not bi_in_range:
      return ordering_compare(abank, bbank);
    }

  } // namespace detail


  /**
   * \brief Comparison operator for library-defined \ref coordinates::pattern objects
   * \todo Streamline this to avoid re-calculating prefix status
   */
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<pattern A, pattern B>
  constexpr std::partial_ordering
  operator<=>(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b)) return values::to_number(get_dimension(a)) <=> values::to_number(get_dimension(b));
    if (get_dimension(a) == 0) return std::partial_ordering::less;
    if (get_dimension(b) == 0) return std::partial_ordering::greater;

    if constexpr (descriptor<A> and descriptor<B>)
    {
      if (internal::get_hash_code(a) == internal::get_hash_code(b)) return std::partial_ordering::equivalent;
      return std::partial_ordering::unordered;
    }
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>)
    {
      if constexpr (descriptor<A>)
        return detail::compare_fixed(OpenKalman::views::single(a), b);
      else if constexpr (descriptor<B>)
        return detail::compare_fixed(a, OpenKalman::views::single(b));
      else
        return detail::compare_fixed(a, b);
    }
    else
    {
      if constexpr (descriptor<A>)
        return detail::compare_impl(OpenKalman::views::single(a), b);
      else if constexpr (descriptor<B>)
        return detail::compare_impl(a, OpenKalman::views::single(b));
      else
        return detail::compare_impl(a, b);
    }
  }


  /**
   * Equality operator for library-defined \ref coordinates::pattern objects
   */
  template<pattern A, pattern B>
  constexpr bool
  operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }


#else
  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b)) return values::to_number(get_dimension(a)) == values::to_number(get_dimension(b));

    if constexpr (descriptor<A> and descriptor<B>) return internal::get_hash_code(a) == internal::get_hash_code(b);
    else if constexpr (descriptor<A>) return std::array {a} == b;
    else if constexpr (descriptor<B>) return a == std::array {b};
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>) return detail::compare_fixed(a, b) == detail::ordering::equivalent;
    else return detail::compare_impl(a, b) == detail::ordering::equivalent;
  }


  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b)) return values::to_number(get_dimension(a)) < values::to_number(get_dimension(b));
    if (get_dimension(a) == 0) return true;

    if constexpr ((descriptor<A> and descriptor<B>) or descriptor<B>) return false;
    else if constexpr (descriptor<A>) return std::array {a} < b;
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>) return detail::compare_fixed(a, b) == detail::ordering::less;
    else return detail::compare_impl(a, b) == detail::ordering::less;
  }


  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b)) return values::to_number(get_dimension(a)) > values::to_number(get_dimension(b));
    if (get_dimension(b) == 0) return true;

    if constexpr ((descriptor<A> and descriptor<B>) or descriptor<A>) return false;
    else if constexpr (descriptor<B>) return a > std::array {b};
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>) return detail::compare_fixed(a, b) == detail::ordering::greater;
    else return detail::compare_impl(a, b) == detail::ordering::greater;
  }


  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b)) return values::to_number(get_dimension(a)) <= values::to_number(get_dimension(b));
    if (get_dimension(a) == 0) return true;

    if constexpr (descriptor<A> and descriptor<B>) return internal::get_hash_code(a) == internal::get_hash_code(b);
    else if constexpr (descriptor<A>) return std::array {a} <= b;
    else if constexpr (descriptor<B>) return a == std::array {b};
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>)
      { auto comp = detail::compare_fixed(a, b); return comp == detail::ordering::less or comp == detail::ordering::equivalent; }
    else
      { auto comp = detail::compare_impl(a, b); return comp == detail::ordering::less or comp == detail::ordering::equivalent; }
  }


  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b)) return values::to_number(get_dimension(a)) >= values::to_number(get_dimension(b));
    if (get_dimension(b) == 0) return true;

    if constexpr (descriptor<A> and descriptor<B>) return internal::get_hash_code(a) == internal::get_hash_code(b);
    else if constexpr (descriptor<B>) return a >= std::array {b};
    else if constexpr (descriptor<A>) return std::array {a} == b;
    else if constexpr (fixed_pattern<A> and fixed_pattern<B>)
      { auto comp = detail::compare_fixed(a, b); return comp == detail::ordering::greater or comp == detail::ordering::equivalent; }
    else
      { auto comp = detail::compare_impl(a, b); return comp == detail::ordering::greater or comp == detail::ordering::equivalent; }
  }


  template<typename A, typename B, std::enable_if_t<pattern<A> and pattern<B>, int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }
#endif


} // namespace OpenKalman::coordinates


#endif //OPENKALMAN_COORDINATES_COMPARISON_OPERATORS_HPP
