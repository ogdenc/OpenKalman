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
 * \brief Comparison operators for \rev vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_COMPARISON_OPERATORS_HPP
#define OPENKALMAN_COMPARISON_OPERATORS_HPP

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif
#include <type_traits>

#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "get_collection_of.hpp"
#include "get_type_index.hpp"
#include "get_vector_space_descriptor_is_euclidean.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"

namespace OpenKalman::descriptor
{
  namespace detail
  {
    namespace ordering
    {
#ifdef __cpp_impl_three_way_comparison
      constexpr auto equivalent = std::partial_ordering::equivalent;
      constexpr auto greater = std::partial_ordering::greater;
      constexpr auto less = std::partial_ordering::less;
      constexpr auto unordered = std::partial_ordering::unordered;


      template<typename A, typename B>
      constexpr std::partial_ordering
      compare(const A& a, const B& b)
      {
        using C = std::common_type_t<A, B>;
        return static_cast<C>(a) <=> static_cast<C>(b);
      }
#else
      constexpr auto equivalent = 0;
      constexpr auto greater = +1;
      constexpr auto less = -1;
      constexpr auto unordered = +2;


      template<typename A, typename B>
      constexpr auto
      compare(const A& a, const B& b)
      {
        using C = std::common_type_t<A, B>;
        return static_cast<C>(a) < static_cast<C>(b) ? less : static_cast<C>(a) > static_cast<C>(b) ? greater : equivalent;
      }
#endif
    } // namespace ordering


    template<std::size_t ix = 0, typename TupA, typename TupB>
    constexpr auto
    compare_fixed_impl(const TupA& tup_a, const TupB& tup_b)
    {
      constexpr auto sA = std::tuple_size_v<TupA>;
      constexpr auto sB = std::tuple_size_v<TupB>;
      if constexpr (ix >= sA and ix >= sB) return ordering::equivalent;
      else if constexpr (ix >= std::tuple_size_v<TupA>) return ordering::less;
      else if constexpr (ix >= std::tuple_size_v<TupB>) return ordering::greater;
      else
      {
        using Ax = std::tuple_element_t<ix, TupA>;
        using Bx = std::tuple_element_t<ix, TupB>;
        if constexpr (std::is_same_v<Ax, Bx>) return compare_fixed_impl<ix + 1>(tup_a, tup_b);
        else if constexpr (not euclidean_vector_space_descriptor<Ax> or not euclidean_vector_space_descriptor<Bx>) return detail::ordering::unordered;
        else if constexpr (ix + 1 >= sA and dimension_size_of_v<Ax> < dimension_size_of_v<Bx>) return detail::ordering::less;
        else if constexpr (ix + 1 >= sB and dimension_size_of_v<Ax> > dimension_size_of_v<Bx>) return detail::ordering::greater;
        else return detail::ordering::unordered;
      }
    }


    template<bool reverse = false, std::size_t ix = 0, typename ItA, typename EndA, typename TupB>
    constexpr auto
    compare_fixed_b_impl(ItA ita, EndA enda, const TupB& tupb)
    {
      static_assert(ix <= std::tuple_size_v<TupB>);
      if constexpr (ix == std::tuple_size_v<TupB>)
      {
        if (ita == enda) return detail::ordering::equivalent;
        if (reverse) return detail::ordering::less;
        return detail::ordering::greater;
      }
      else
      {
        using B = std::tuple_element_t<ix, TupB>;
        if constexpr (descriptor::euclidean_vector_space_descriptor<B>)
        {
          constexpr std::size_t N = descriptor::dimension_size_of_v<B>;
          std::size_t n = 0;
          for (; ita != enda and descriptor::get_vector_space_descriptor_is_euclidean(*ita); ++ita)
            n += descriptor::get_dimension_size_of(*ita);
          if (n == N) return compare_fixed_b_impl<reverse, ix + 1>(ita, enda, tupb);
          if (ita == enda)
            return reverse == (n < N) ? detail::ordering::greater : detail::ordering::less;
          else return detail::ordering::unordered;
        }
        else
        {
          if (ita == enda) return reverse ? detail::ordering::greater : detail::ordering::less;
          if (descriptor::get_type_index(*ita) != descriptor::get_type_index(std::get<ix>(tupb)))
            return detail::ordering::unordered;
          return compare_fixed_b_impl<reverse, ix + 1>(++ita, enda, tupb);
        }
      }
    }


    template<typename A, typename B>
    constexpr auto
    compare_impl(const A& a, const B& b)
    {
      if constexpr (dimension_size_of_v<A> == 0 or dimension_size_of_v<B> == 0 or
        (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>))
      {
        return ordering::compare(get_dimension_size_of(a), get_dimension_size_of(b));
      }
      else if constexpr (static_vector_space_descriptor<A> and static_vector_space_descriptor<B>)
      {
        auto coll_a = descriptor::get_collection_of(a);
        auto coll_b = descriptor::get_collection_of(b);
        return detail::compare_fixed_impl(coll_a, coll_b);
      }
      else if constexpr (static_vector_space_descriptor<B>)
      {
        if constexpr (euclidean_vector_space_descriptor<A>)
        {
          auto b0 = std::get<0>(descriptor::get_collection_of(b));
          if (get_vector_space_descriptor_is_euclidean(b0))
          {
            if (vector_space_component_count_v<B> <= 1)
              return detail::ordering::compare(get_dimension_size_of(a), get_dimension_size_of(b0));
            if (get_dimension_size_of(a) <= get_dimension_size_of(b0)) return detail::ordering::less;
            return detail::ordering::unordered;
          }
          return detail::ordering::unordered;
        }
        else
        {
          auto coll_a = descriptor::get_collection_of(a);
#ifdef __cpp_lib_ranges
          auto ita = std::ranges::begin(coll_a);
          auto enda = std::ranges::end(coll_a);
#else
          using std::begin, std::end;
          auto ita = begin(coll_a);
          auto enda = end(coll_a);
#endif
          return detail::compare_fixed_b_impl<false>(ita, enda, descriptor::get_collection_of(b));
        }
      }
      else if constexpr (static_vector_space_descriptor<A>)
      {
        if constexpr (euclidean_vector_space_descriptor<B>)
        {
          auto a0 = std::get<0>(descriptor::get_collection_of(a));
          if (get_vector_space_descriptor_is_euclidean(a0))
          {
            if (vector_space_component_count_v<A> <= 1)
              return detail::ordering::compare(get_dimension_size_of(a0), get_dimension_size_of(b));
            if (get_dimension_size_of(a0) >= get_dimension_size_of(b)) return detail::ordering::greater;
            return detail::ordering::unordered;
          }
          return detail::ordering::unordered;
        }
        else
        {
          auto coll_b = descriptor::get_collection_of(b);
#ifdef __cpp_lib_ranges
          auto itb = std::ranges::begin(coll_b);
          auto endb = std::ranges::end(coll_b);
#else
          using std::begin, std::end;
          auto itb = begin(coll_b);
          auto endb = end(coll_b);
#endif
          return detail::compare_fixed_b_impl<true>(itb, endb, descriptor::get_collection_of(a));
        }
      }
      else
      {
        auto coll_a = descriptor::get_collection_of(a);
        auto coll_b = descriptor::get_collection_of(b);
  #ifdef __cpp_lib_ranges
        auto ita = std::ranges::begin(coll_a);
        auto enda = std::ranges::end(coll_a);
        auto itb = std::ranges::begin(coll_b);
        auto endb = std::ranges::end(coll_b);
  #else
        using std::begin, std::end;
        auto ita = begin(coll_a);
        auto enda = end(coll_a);
        auto itb = begin(coll_b);
        auto endb = end(coll_b);
  #endif
        for (; ita != enda or itb != endb; ++ita, ++itb)
        {
          std::size_t ni = 0, nj = 0;
          for (; ita != enda and descriptor::get_vector_space_descriptor_is_euclidean(*ita); ++ita)
            ni += descriptor::get_dimension_size_of(*ita);
          for (; itb != endb and descriptor::get_vector_space_descriptor_is_euclidean(*itb); ++itb)
            nj += descriptor::get_dimension_size_of(*itb);
          if (ita == enda)
          {
            if (itb == endb) return detail::ordering::compare(ni, nj);
            return ni <= nj ? detail::ordering::less : detail::ordering::unordered;
          }
          if (itb == endb)
            return ni >= nj ? detail::ordering::greater : detail::ordering::unordered;
          if (descriptor::get_type_index(*ita) != descriptor::get_type_index(*itb))
            return detail::ordering::unordered;
        }
        return detail::ordering::equivalent;
      }
    }

  } // namespace detail


/**
   * \brief Comparison operator for library-defined \ref vector_space_descriptor objects
   * \details Comparison of dynamic non-euclidean descriptors is defined elsewhere.
   * \todo Streamline this to avoid re-calculating prefix status
   */
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr auto operator<=>(const A& a, const B& b)
  {
    return detail::compare_impl(a, b);
  }


  /**
   * Equality operator for library-defined \ref vector_space_descriptor objects
   */
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr bool operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }
#else
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    return detail::compare_impl(a, b) == detail::ordering::equivalent;
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    return detail::compare_impl(a, b) == detail::ordering::less;
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    return detail::compare_impl(a, b) == detail::ordering::greater;
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return operator<(a, b) or operator==(a, b);
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    return operator>(a, b) or operator==(a, b);
  }
#endif


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_COMPARISON_OPERATORS_HPP
