/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Functions for \ref is_prefix.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_IS_PREFIX_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_IS_PREFIX_HPP

#include <type_traits>
#include <basics/utils.hpp>
#include <linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp>
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "canonical_equivalent.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_hash_code.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/prefix_base_of.hpp"

namespace OpenKalman::descriptor::internal
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename A, typename B, typename = void>
    struct is_prefix_impl_fixed : std::false_type {};

    template<typename A, typename B>
    struct is_prefix_impl_fixed<A, B, std::void_t<typename descriptor::internal::prefix_base_of<A, B>::type>>
      : std::true_type {};
#endif


    template<std::size_t ix, typename TupA, typename ItB, typename EndB>
    constexpr auto
    is_prefix_impl_a(const TupA&, ItB, EndB) { return true; }


#ifdef __cpp_concepts
    template<std::size_t ix, typename TupA, typename ItB, typename EndIt> requires (ix < std::tuple_size_v<TupA>)
#else
    template<std::size_t ix, typename TupA, typename ItB, typename EndIt, std::enable_if_t<(ix < std::tuple_size_v<TupA>), int> = 0>
#endif
    constexpr auto
    is_prefix_impl_a(const TupA& tupa, ItB itb, EndIt endb)
    {
      using A = std::tuple_element_t<ix, TupA>;
      if constexpr (descriptor::euclidean_vector_space_descriptor<A>)
      {
        constexpr std::size_t N = descriptor::dimension_size_of_v<A>;
        std::size_t n = 0;
        for (; itb != endb and descriptor::get_vector_space_descriptor_is_euclidean(*itb); ++itb)
          n += descriptor::get_dimension_size_of(*itb);
        if (itb == endb or n != N) return false;
        return is_prefix_impl_a<ix + 1>(tupa, ++itb, endb);
      }
      else
      {
        if (itb == endb)
          return false;
        if (descriptor::get_hash_code(*itb) == descriptor::get_hash_code(std::get<ix>(tupa)))
          return is_prefix_impl_a<ix + 1>(tupa, ++itb, endb);
        else
          return false;
      }
    }



    template<std::size_t ix, typename ItA, typename EndA, typename TupB>
    constexpr auto
    is_prefix_impl_b(ItA ita, EndA enda, const TupB& tupb)
    {
      std::cout << "c0" << std::endl;
      static_assert(ix <= std::tuple_size_v<TupB>);
      if (ita == enda) return true;
      else return descriptor::get_dimension_size_of(*ita) == 0 and is_prefix_impl_b<ix>(++ita, enda, tupb);
    }


#ifdef __cpp_concepts
    template<std::size_t ix, typename ItA, typename EndA, typename TupB> requires (ix < std::tuple_size_v<TupB>)
#else
    template<std::size_t ix, typename ItA, typename EndA, typename TupB, std::enable_if_t<(ix < std::tuple_size_v<TupB>), int> = 0>
#endif
    constexpr auto
    is_prefix_impl_b(ItA ita, EndA enda, const TupB& tupb)
    {
      using B = std::tuple_element_t<ix, TupB>;
      if constexpr (descriptor::euclidean_vector_space_descriptor<B>)
      {
        std::cout << "a0" << std::endl;
        constexpr std::size_t N = descriptor::dimension_size_of_v<B>;
        std::size_t n = 0;
        for (; ita != enda and descriptor::get_vector_space_descriptor_is_euclidean(*ita); ++ita)
          n += descriptor::get_dimension_size_of(*ita);
        std::cout << "n: " << n << " vs " << "N:" << N << std::endl;
        if (ita == enda) return n <= N;
        std::cout << "a01" << std::endl;
        return n == N and is_prefix_impl_b<ix + 1>(++ita, enda, tupb);
      }
      else
      {
        std::cout << "a1" << std::endl;
        if (ita == enda) return true;
        std::cout << "a2" << std::endl;
        std::cout << descriptor::get_hash_code(*ita) << " vs " << descriptor::get_hash_code(std::get<ix>(tupb));
        if (descriptor::get_hash_code(*ita) != descriptor::get_hash_code(std::get<ix>(tupb))) return false;
        std::cout << "a3" << std::endl;
        return is_prefix_impl_b<ix + 1>(++ita, enda, tupb);
      }
    }

  }


  /**
   * \internal
   * \brief Whether <code>a</code> is a prefix of <code>b</code>.
   * \details A is a prefix of B if A matches at least the initial portion of B (including if A is equivalent to B).
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr std::convertible_to<bool> auto
#else
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr auto
#endif
  is_prefix(const A& a, const B& b)
  {
    if constexpr (dimension_size_of_v<A> == 0)
    {
      return std::true_type{};
    }
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
    {
      return value::operation {std::less_equal<>{}, get_dimension_size_of(a), get_dimension_size_of(b)};
    }
    else if constexpr (euclidean_vector_space_descriptor<B>)
    {
      return value::operation {
        std::logical_and<>{},
        get_vector_space_descriptor_is_euclidean(a),
        value::operation{std::less_equal<>{}, get_dimension_size_of(a), get_dimension_size_of(b)}};
    }
    else if constexpr (euclidean_vector_space_descriptor<A>)
    {
      if constexpr (static_vector_space_descriptor<B>)
      {
        auto b0 = std::get<0>(descriptor::get_collection_of(canonical_equivalent(b)));
        return value::operation {
          std::logical_and<>{},
          get_vector_space_descriptor_is_euclidean(b0),
          value::operation{std::less_equal<>{}, get_dimension_size_of(a), get_dimension_size_of(b0)}};
      }
      else
      {
#ifdef __cpp_lib_ranges
        using std::begin, std::end;
        auto itb = std::ranges::begin(b);
        auto endb = std::ranges::end(b);
#else
        auto itb = begin(b);
        auto endb = end(b);
#endif
        if (itb == endb) return descriptor::get_dimension_size_of(a) == 0;
        else
        {
          std::size_t n = 0;
          for (; itb != endb and descriptor::get_vector_space_descriptor_is_euclidean(*itb); ++itb)
            n += descriptor::get_dimension_size_of(*itb);
          return descriptor::get_dimension_size_of(a) <= n;
        }
      }
    }
    else if constexpr (static_vector_space_descriptor<A> and static_vector_space_descriptor<B>)
    {
      using CA = std::decay_t<decltype(canonical_equivalent(a))>;
      using CB = std::decay_t<decltype(canonical_equivalent(b))>;
#ifdef __cpp_concepts
      return std::bool_constant<requires { typename descriptor::internal::prefix_base_of_t<CA, CB>; }>{};
#else
      return detail::is_prefix_impl_fixed<CA, CB>::value;
#endif
    }
    else if constexpr (static_vector_space_descriptor<A>)
    {
      auto coll_b = descriptor::get_collection_of(canonical_equivalent(b));
#ifdef __cpp_lib_ranges
      return detail::is_prefix_impl_a<0>(descriptor::get_collection_of(a), std::ranges::begin(coll_b), std::ranges::end(coll_b));
#else
      using std::begin, std::end;
      return detail::is_prefix_impl_a<0>(descriptor::get_collection_of(a), begin(coll_b), end(coll_b));
#endif
    }
    else if constexpr (static_vector_space_descriptor<B>)
    {
        std::cout << "----" << std::endl;
      auto coll_a = descriptor::get_collection_of(canonical_equivalent(a));
#ifdef __cpp_lib_ranges
      return detail::is_prefix_impl_b<0>(std::ranges::begin(coll_a), std::ranges::end(coll_a), descriptor::get_collection_of(b));
#else
      using std::begin, std::end;
      return detail::is_prefix_impl_b<0>(begin(coll_a), end(coll_a), descriptor::get_collection_of(b));
#endif
    }
    else
    {
#ifdef __cpp_lib_ranges
      auto ita = std::ranges::begin(a);
      auto enda = std::ranges::end(a);
      auto itb = std::ranges::begin(b);
      auto endb = std::ranges::end(b);
#else
      using std::begin, std::end;
      auto ita = begin(a);
      auto enda = end(a);
      auto itb = begin(b);
      auto endb = end(b);
#endif
      for (; ita != enda or itb != endb; ++ita, ++itb)
      {
        std::size_t nj = 0, ni = 0;
        for (; ita != enda and descriptor::get_vector_space_descriptor_is_euclidean(*ita); ++ita)
          nj += descriptor::get_dimension_size_of(*ita);
        for (; itb != endb and descriptor::get_vector_space_descriptor_is_euclidean(*itb); ++itb)
          ni += descriptor::get_dimension_size_of(*itb);
        if (ita == enda) return nj <= ni;
        if (itb == endb or nj != ni) return false;
        if (descriptor::get_hash_code(*ita) != descriptor::get_hash_code(*itb)) return false;
      }
      return true;
    }
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_IS_PREFIX_HPP
