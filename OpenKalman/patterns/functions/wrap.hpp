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
 * \brief Definition for \ref patterns::wrap.
 */

#ifndef OPENKALMAN_PATTERNS_WRAP_HPP
#define OPENKALMAN_PATTERNS_WRAP_HPP

#include <type_traits>
#include <functional>
#include "collections/collections.hpp"
#include "patterns/interfaces/pattern_descriptor_traits.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/concepts/descriptor_collection.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/functions/to_stat_space.hpp"
#include "patterns/functions/from_stat_space.hpp"
#include "patterns/traits/dimension_of.hpp"

namespace OpenKalman::patterns
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Traits, typename = void>
    struct wrap_trait_defined_for : std::false_type {};

    template<typename Traits>
    struct wrap_trait_defined_for<Traits, std::void_t<decltype(Traits::wrap)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief wraps a range reflecting vector-space data to its primary range.
   * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
   * or in other words, performing <code>to_stat_space</code> followed by <code>from_stat_space<code>.
   * \param t A \ref patterns::descriptor "descriptor".
   * \param data_view A range within a data object corresponding to \ref patterns::descriptor "descriptor" t.
   */
#ifdef __cpp_concepts
  template<descriptor T, collections::collection R>
  constexpr collections::collection decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  wrap(const T& t, R&& data_view)
  {
    if constexpr (dimension_of_v<T> != stdex::dynamic_extent and collections::size_of_v<R> != stdex::dynamic_extent)
      static_assert(dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (euclidean_pattern<T>)
    {
      return std::forward<R>(data_view);
    }
    else
    {
      using U = std::decay_t<stdex::unwrap_ref_decay_t<T>>;
      using Traits = interface::pattern_descriptor_traits<U>;
#ifdef __cpp_concepts
      if constexpr (requires { Traits::wrap; })
#else
      if constexpr (detail::wrap_trait_defined_for<Traits>::value)
#endif
      {
        if constexpr (std::is_same_v<U, T>)
          return stdex::invoke(Traits::wrap, t, std::forward<R>(data_view));
        else
          return stdex::invoke(Traits::wrap, t.get(), std::forward<R>(data_view));
      }
      else
      {
        return from_stat_space(t, to_stat_space(t, std::forward<R>(data_view)));
      }
    }
  }


  namespace detail
  {
    template<std::size_t t_i = 0, std::size_t data_view_i = 0, typename T, typename R, typename...Out>
    static constexpr auto
    wrap_tuple(const T& t, const R& data_view, Out...out)
    {
      if constexpr (t_i < collections::size_of_v<T>)
      {
        decltype(auto) t_elem = collections::get<t_i>(t);
        auto dim = get_dimension(t_elem);
        auto data_view_sub = collections::views::slice(data_view, std::integral_constant<std::size_t, data_view_i>{}, dim);
        auto o = wrap(t_elem, std::move(data_view_sub));
        return wrap_tuple<t_i + 1, data_view_i + values::fixed_value_of_v<decltype(dim)>>(t, data_view, std::move(out)..., std::move(o));
      }
      else
      {
        return collections::views::concat(std::move(out)...);
      }
    }

  }


  /**
   * \overload
   * \brief wraps a range reflecting vector-space data to its primary range.
   * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
   * or in other words, performing <code>to_stat_space</code> followed by <code>from_stat_space<code>.
   * \param t A \ref patterns::descriptor_collection "descriptor_collection".
   * \param data_view A range within a data object corresponding to \ref patterns::descriptor_collection "descriptor_collection" t.
   */
#ifdef __cpp_concepts
  template<descriptor_collection T, collections::collection R>
  constexpr collections::collection decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor_collection<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  wrap(const T& t, R&& data_view)
  {
    if constexpr (dimension_of_v<T> != stdex::dynamic_extent and collections::size_of_v<R> != stdex::dynamic_extent)
      static_assert(dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (dimension_of_v<T> == stdex::dynamic_extent)
    {
      std::vector<collections::common_collection_type_t<R>> data;
      data.reserve(get_dimension(t));
      std::size_t i = 0;
      for (auto& d : t)
      {
        auto dim = get_dimension(d);
#if __cpp_lib_containers_ranges >= 202002L
        data.append_range(wrap(d, collections::views::slice(data_view, i, dim)));
#else
        auto sd = wrap(d, collections::views::slice(data_view, i, dim));
        data.insert(data.end(), sd.cbegin(), sd.cend());
#endif
        i += values::to_value_type(dim);
      }
      return data;
    }
    else //if constexpr (fixed_pattern<T>)
    {
      return detail::wrap_tuple(t, data_view);
    }
  }

}


#endif