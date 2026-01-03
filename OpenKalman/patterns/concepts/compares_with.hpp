/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref compares_with.
 */

#ifndef OPENKALMAN_COORDINATE_COMPARES_WITH_HPP
#define OPENKALMAN_COORDINATE_COMPARES_WITH_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern.hpp"

namespace OpenKalman::patterns
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


    constexpr std::size_t
    inc_bank(std::size_t bank, std::size_t inc)
    {
      return (bank == stdex::dynamic_extent or inc == stdex::dynamic_extent) ? stdex::dynamic_extent : bank + inc;
    }


#ifdef __cpp_concepts
    template<typename A, typename B>
#else
    template<typename A, typename B, typename = void>
#endif
    struct same_descriptors : std::false_type {};

#ifdef __cpp_concepts
    template<typename A, typename B> requires
      std::bool_constant<internal::get_descriptor_hash_code(A{}) == internal::get_descriptor_hash_code(B{})>::value
    struct same_descriptors<A, B>
#else
    template<typename A, typename B>
    struct same_descriptors<A, B, std::enable_if_t<
      std::bool_constant<internal::get_descriptor_hash_code(A{}) == internal::get_descriptor_hash_code(B{})>::value>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename A, typename B>
#else
    template<typename A, typename B, typename = void>
#endif
    struct different_descriptors : std::false_type {};

#ifdef __cpp_concepts
    template<typename A, typename B> requires
      std::bool_constant<internal::get_descriptor_hash_code(A{}) != internal::get_descriptor_hash_code(B{})>::value
    struct different_descriptors<A, B>
#else
    template<typename A, typename B>
    struct different_descriptors<A, B, std::enable_if_t<
      std::bool_constant<internal::get_descriptor_hash_code(A{}) != internal::get_descriptor_hash_code(B{})>::value>>
#endif
      : std::true_type {};


    template<typename A, typename B, auto comp, applicability app,
      std::size_t ia = 0, std::size_t ib = 0, std::size_t abank = 0, std::size_t bbank = 0>
    constexpr bool
    compares_with_iter()
    {
      using stdex::partial_ordering;

      if constexpr (ia < collections::size_of_v<A>)
      {
        using Ai = std::decay_t<collections::collection_element_t<ia, A>>;
        constexpr bool ae = euclidean_pattern<Ai>;
        constexpr std::size_t dim_Ai = dimension_of_v<Ai>;
        if constexpr (ib < collections::size_of_v<B>)
        {
          using Bi = std::decay_t<collections::collection_element_t<ib, B>>;
          constexpr bool be = euclidean_pattern<Bi>;
          constexpr std::size_t dim_Bi = dimension_of_v<Bi>;

          if constexpr (ae and be)
          {
            return compares_with_iter<A, B, comp, app, ia + 1, ib + 1, inc_bank(abank, dim_Ai), bbank + dim_Bi>();
          }
          else if constexpr (ae)
          {
            return compares_with_iter<A, B, comp, app, ia + 1, ib, inc_bank(abank, dim_Ai), bbank>();
          }
          else if constexpr (be)
          {
            return compares_with_iter<A, B, comp, app, ia, ib + 1, abank, inc_bank(bbank, dim_Bi)>();
          }
          else if constexpr (abank == stdex::dynamic_extent or bbank == stdex::dynamic_extent)
          {
            return app == applicability::permitted and do_comps<comp>(partial_ordering::equivalent, partial_ordering::unordered);
          }
          else if constexpr (abank != bbank)
          {
            return do_comps<comp>(partial_ordering::unordered);
          }
          else if constexpr (same_descriptors<Ai, Bi>::value)
          {
            return compares_with_iter<A, B, comp, app, ia + 1, ib + 1>();
          }
          else if constexpr (different_descriptors<Ai, Bi>::value)
          {
            return do_comps<comp>(partial_ordering::unordered);
          }
          else if constexpr (do_comps<comp>(partial_ordering::equivalent))
          {
            return app == applicability::permitted and
              compares_with_iter<A, B, comp, app, ia + 1, ib + 1>();
          }
          else if constexpr (do_comps<comp>(partial_ordering::unordered))
          {
            return app == applicability::permitted or
              compares_with_iter<A, B, comp, app, ia + 1, ib + 1>();
          }
          else if constexpr (do_comps<comp>(partial_ordering::less))
          {
            return app == applicability::permitted and
              (dim_Ai == stdex::dynamic_extent or dim_Ai == 0) and
              compares_with_iter<A, B, comp, app, ia + 1, ib + 1, dim_Ai, dim_Bi>();
          }
          else //if constexpr (do_comps<comp>(partial_ordering::greater))
          {
            return app == applicability::permitted and
              (dim_Bi == stdex::dynamic_extent or dim_Bi == 0) and
              compares_with_iter<A, B, comp, app, ia + 1, ib + 1, dim_Ai, dim_Bi>();
          }
        }
        else if constexpr (euclidean_pattern<Ai>)
        {
          return compares_with_iter<A, B, comp, app, ia + 1, ib, inc_bank(abank, dim_Ai), bbank>();
        }
        else if constexpr (abank == stdex::dynamic_extent or bbank == stdex::dynamic_extent)
        {
          return (dim_Ai != stdex::dynamic_extent and do_comps<comp>(partial_ordering::unordered)) or
            (app == applicability::permitted and (dim_Ai == stdex::dynamic_extent or do_comps<comp>(partial_ordering::greater)));
        }
        else if (abank >= bbank)
        {
          return do_comps<comp>(partial_ordering::greater);
        }
        else if constexpr (dim_Ai == stdex::dynamic_extent)
        {
          return app == applicability::permitted and
            ( do_comps<comp>(partial_ordering::unordered, partial_ordering::greater, partial_ordering::equivalent) or
              (do_comps<comp>(partial_ordering::less) and bbank > 0));
        }
        else
        {
          return do_comps<comp>(partial_ordering::unordered);
        }
      }
      else if constexpr (ib < collections::size_of_v<B>) // ia >= collections::size_of_v<A>
      {
        using Bi = std::decay_t<collections::collection_element_t<ib, B>>;
        constexpr std::size_t dim_Bi = dimension_of_v<Bi>;

        if constexpr (euclidean_pattern<Bi>)
        {
          return compares_with_iter<A, B, comp, app, ia, ib + 1, abank, inc_bank(bbank, dim_Bi)>();
        }
        else if constexpr (abank == stdex::dynamic_extent or bbank == stdex::dynamic_extent)
        {
          return (dim_Bi != stdex::dynamic_extent and do_comps<comp>(partial_ordering::unordered)) or
            (app == applicability::permitted and (dim_Bi == stdex::dynamic_extent or do_comps<comp>(partial_ordering::less)));
        }
        else if (abank <= bbank)
        {
          return do_comps<comp>(partial_ordering::less);
        }
        else if constexpr (dim_Bi == stdex::dynamic_extent)
        {
          return app == applicability::permitted and
            ( do_comps<comp>(partial_ordering::unordered, partial_ordering::less, partial_ordering::equivalent) or
              (do_comps<comp>(partial_ordering::greater) and abank > 0));
        }
        else
        {
          return do_comps<comp>(partial_ordering::unordered);
        }
      }
      else if constexpr (abank != stdex::dynamic_extent and bbank != stdex::dynamic_extent)
      {
        return do_comps<comp>(stdex::compare_three_way{}(abank, bbank));
      }
      else if constexpr (abank == 0)
      {
        return do_comps<comp, app == applicability::guaranteed>(partial_ordering::less, partial_ordering::equivalent);
      }
      else if constexpr (bbank == 0)
      {
        return do_comps<comp, app == applicability::guaranteed>(partial_ordering::greater, partial_ordering::equivalent);
      }
      else if constexpr (abank != stdex::dynamic_extent or bbank != stdex::dynamic_extent)
      {
        return app == applicability::permitted and do_comps<comp>(partial_ordering::less, partial_ordering::greater, partial_ordering::equivalent);
      }
      else
      {
        return app == applicability::permitted;
      }
    }


    template<typename T, typename U, auto comp, applicability a>
    constexpr bool
    compares_with_impl()
    {
      ///
      /// -- Either T or U is a descriptor
      ///

      if constexpr (descriptor<T>)
      {
        return compares_with_impl<std::tuple<stdex::unwrap_ref_decay_t<T>>, U, comp, a>();
      }
      else if constexpr (descriptor<U>)
      {
        return compares_with_impl<T, std::tuple<stdex::unwrap_ref_decay_t<U>>, comp, a>();
      }

      ///
      /// -- Either T or U has no dimensions (e.g., they are unsized collections)
      ///

      else if constexpr (not values::fixed<dimension_of<T>> and not values::fixed<dimension_of<U>>)
      {
        using DT = stdex::ranges::range_value_t<T>;
        using DU = stdex::ranges::range_value_t<U>;
        if constexpr ((euclidean_pattern<DT> and euclidean_pattern<DU>))
          return detail::do_comps<comp>(stdex::partial_ordering::equivalent);
        else if constexpr (dimension_of_v<DT> == 0 or dimension_of_v<DU> == 0)
          return values::size_compares_with<dimension_of<DT>, dimension_of<DU>, comp, a>;
        else if constexpr (dimension_of_v<DT> == stdex::dynamic_extent or dimension_of_v<DU> == stdex::dynamic_extent)
          return a == applicability::permitted;
        else if constexpr (compares_with_impl<DT, DU, &stdex::is_eq, a>())
          return detail::do_comps<comp>(stdex::partial_ordering::equivalent);
        else
          return false;
      }
      else if constexpr (not values::fixed<dimension_of<T>>) // and values::fixed<dimension_of<U>>
      {
        using DT = stdex::ranges::range_value_t<T>;
        if constexpr (euclidean_pattern<DT> and euclidean_pattern<U>)
        {
          return detail::do_comps<comp>(stdex::partial_ordering::greater);
        }
        else if constexpr (dimension_of_v<U> == 0)
        {
          if constexpr (dimension_of_v<DT> == stdex::dynamic_extent)
            return detail::do_comps<comp, a == applicability::guaranteed>(
              stdex::partial_ordering::greater, stdex::partial_ordering::equivalent);
          else
            return detail::do_comps<comp>(stdex::partial_ordering::greater);
        }
        else if constexpr (dimension_of_v<DT> == stdex::dynamic_extent)
        {
          return a == applicability::permitted;
        }
        else if constexpr (dimension_of_v<U> == stdex::dynamic_extent)
        {
          if constexpr (detail::do_comps<comp>(stdex::partial_ordering::unordered))
            return true;
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::greater))
            return compares_with_impl<DT, stdex::ranges::range_value_t<U>, &stdex::is_eq, a>();
          else
            return false;
        }
        else // U has a fixed dimension and size
        {
          return compares_with_impl<collections::repeat_tuple_view<collections::size_of_v<U> + 1, DT>, U, comp, a>();
        }
      }
      else if constexpr (not values::fixed<dimension_of<U>>) // and values::fixed<dimension_of<T>>
      {
        using DU = stdex::ranges::range_value_t<U>;
        if constexpr (euclidean_pattern<T> and euclidean_pattern<DU>)
        {
          return detail::do_comps<comp>(stdex::partial_ordering::less);
        }
        else if constexpr (not values::fixed<dimension_of<DU>>)
        {
          return a == applicability::permitted;
        }
        else if constexpr (dimension_of_v<T> == 0)
        {
          if constexpr (dimension_of_v<DU> == stdex::dynamic_extent)
            return detail::do_comps<comp, a == applicability::guaranteed>(
              stdex::partial_ordering::less, stdex::partial_ordering::equivalent);
          else
            return detail::do_comps<comp>(stdex::partial_ordering::less);
        }
        else if constexpr (dimension_of_v<DU> == stdex::dynamic_extent)
        {
          return a == applicability::permitted;
        }
        else if constexpr (dimension_of_v<T> == stdex::dynamic_extent)
        {
          if constexpr (detail::do_comps<comp>(stdex::partial_ordering::unordered))
            return true;
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::less))
            return compares_with_impl<stdex::ranges::range_value_t<T>, DU, &stdex::is_eq, a>();
          else
            return false;
        }
        else // T has a fixed dimension and size
        {
          return compares_with_impl<T, collections::repeat_tuple_view<collections::size_of_v<T> + 1, DU>, comp, a>();
        }
      }

      ///
      /// -- Both T and U have dimensions (either fixed or dynamic)
      ///

      else if constexpr (dimension_of_v<T> == 0 and dimension_of_v<U> == 0)
      {
        return detail::do_comps<comp>(stdex::partial_ordering::equivalent);
      }
      else if constexpr (dimension_of_v<T> == 0 and collections::size_of_v<U> == stdex::dynamic_extent)
      {
        return detail::do_comps<comp, a == applicability::guaranteed>(stdex::partial_ordering::less, stdex::partial_ordering::equivalent);
      }
      else if constexpr (dimension_of_v<U> == 0 and collections::size_of_v<T> == stdex::dynamic_extent)
      {
        return detail::do_comps<comp, a == applicability::guaranteed>(stdex::partial_ordering::greater, stdex::partial_ordering::equivalent);
      }
      else if constexpr (collections::size_of_v<T> != stdex::dynamic_extent and collections::size_of_v<U> != stdex::dynamic_extent)
      {
        if constexpr (collections::size_of<T>::value == 0 and collections::size_of<U>::value == 0)
          return detail::do_comps<comp>(stdex::partial_ordering::equivalent);
        else
          return detail::compares_with_iter<T, U, comp, a>();
      }
      else if constexpr (collections::size_of_v<T> != stdex::dynamic_extent) // collections::size_of_v<U> == stdex::dynamic_extent
      {
        using DU = stdex::ranges::range_value_t<U>;
        if constexpr (dimension_of_v<DU> == 0)
        {
          return detail::do_comps<comp>(stdex::partial_ordering::greater);
        }
        else if constexpr (dimension_of_v<T> != stdex::dynamic_extent and dimension_of_v<DU> != stdex::dynamic_extent)
        {
          if constexpr (euclidean_pattern<T> and euclidean_pattern<DU>)
          {
            if constexpr (detail::do_comps<comp>(stdex::partial_ordering::unordered))
              return a == applicability::permitted or dimension_of_v<T> % dimension_of_v<DU> != 0;
            else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::less, stdex::partial_ordering::greater))
              return a == applicability::permitted;
            else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::equivalent))
              return a == applicability::permitted and dimension_of_v<T> % dimension_of_v<DU> == 0;
            else
              return a == applicability::permitted;
          }
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::unordered))
          {
            return a == applicability::permitted or
              compares_with_impl<T, collections::repeat_tuple_view<collections::size_of_v<T>, DU>, comp, a>();
          }
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::greater))
          {
            return a == applicability::permitted;
          }
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::less))
          {
            return a == applicability::permitted and
              compares_with_impl<T, collections::repeat_tuple_view<collections::size_of_v<T>, DU>, &stdex::is_eq, a>();
          }
          else
          {
            return a == applicability::permitted and
              compares_with_impl<T, collections::repeat_tuple_view<collections::size_of_v<T>, DU>, comp, a>();
          }
        }
        else
        {
          return a == applicability::permitted;
        }
      }
      else if constexpr (collections::size_of_v<U> != stdex::dynamic_extent) // collections::size_of_v<T> == stdex::dynamic_extent
      {
        using DT = stdex::ranges::range_value_t<T>;
        if constexpr (dimension_of_v<DT> == 0)
        {
          return detail::do_comps<comp>(stdex::partial_ordering::less);
        }
        else if constexpr (dimension_of_v<DT> != stdex::dynamic_extent and dimension_of_v<U> != stdex::dynamic_extent)
        {
          if constexpr (euclidean_pattern<DT> and euclidean_pattern<U>)
          {
            if constexpr (detail::do_comps<comp>(stdex::partial_ordering::unordered))
              return a == applicability::permitted or dimension_of_v<U> % dimension_of_v<DT> != 0;
            else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::less, stdex::partial_ordering::greater))
              return a == applicability::permitted;
            else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::equivalent))
              return a == applicability::permitted and dimension_of_v<U> % dimension_of_v<DT> == 0;
            else
              return a == applicability::permitted;
          }
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::unordered))
          {
            return a == applicability::permitted or
              compares_with_impl<collections::repeat_tuple_view<collections::size_of_v<U>, DT>, U, comp, a>();
          }
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::less))
          {
            return a == applicability::permitted;
          }
          else if constexpr (detail::do_comps<comp>(stdex::partial_ordering::greater))
          {
            return a == applicability::permitted and
              compares_with_impl<collections::repeat_tuple_view<collections::size_of_v<U>, DT>, U, &stdex::is_eq, a>();
          }
          else
          {
            return a == applicability::permitted and
              compares_with_impl<collections::repeat_tuple_view<collections::size_of_v<U>, DT>, U, comp, a>();
          }
        }
        else
        {
          return a == applicability::permitted;
        }
      }
      else // if constexpr (collections::size_of_v<T> == stdex::dynamic_extent and collections::size_of_v<U> == stdex::dynamic_extent)
      {
        return a == applicability::permitted;
      }
    }

  }


  /**
   * \brief Compares two \ref patterns::pattern objects.
   * \details Every \ref pattern in the set must be potentially comparable to every other respective \ref pattern in the set.
   * Sets of patterns are equivalent if they are treated functionally the same.
   * - Any \ref pattern is equivalent to itself.
   * - std::tuple<As...> is equivalent to std::tuple<Bs...>, if each As is equivalent to its respective Bs.
   * - std::tuple<A> is equivalent to A, and vice versa.
   * - Dynamic \ref patterns::euclidean_pattern objects are equivalent to any other \ref patterns::euclidean_pattern,
   * \par Examples:
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction&gt;&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction, angle::Radians&gt;, less_than<>, applicability::guaranteed&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Dimensions<>, Direction, angle::Radians&gt;, less_than<>, applicability::permitted&gt;</code>
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, typename U, auto comp = &stdex::is_eq, applicability a = applicability::guaranteed>
#ifdef __cpp_concepts
  concept compares_with =
#else
  constexpr bool compares_with =
#endif
    pattern<T> and pattern<U> and
    std::is_invocable_r_v<bool, decltype(comp), stdex::partial_ordering> and
    detail::compares_with_impl<T, U, comp, a>();


}

#endif
