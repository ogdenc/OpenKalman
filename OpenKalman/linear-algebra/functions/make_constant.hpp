/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_constant.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_HPP
#define OPENKALMAN_MAKE_CONSTANT_HPP

#include "linear-algebra/functions/internal/constant_mdspan_policies.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/adapters/pattern_adapter.hpp"
#include "linear-algebra/functions/attach_pattern.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref indexible object in which every element is a constant \ref values::value "value".
   * \returns An mdspan returning the constant value at every set of indices.
   * \param c A \ref values::value "value"
   * \param extents An std::extents object defining the extents.
   */
#ifdef __cpp_concepts
  template<values::value C, typename IndexType, std::size_t...Extents>
  constexpr constant_object auto
#else
  template<typename C, typename IndexType, std::size_t...Extents, std::enable_if_t<values::value<C>, int> = 0>
  constexpr auto
#endif
  make_constant(C c, stdex::extents<IndexType, Extents...> extents)
  {
    using ElementType = C;
    using E = stdex::extents<IndexType, Extents...>;
    auto mapping = typename internal::layout_constant::mapping<E> {extents};
    auto accessor = internal::accessor_constant<ElementType> {std::move(c)};
    return stdex::mdspan {accessor.data_handle(), mapping, accessor};
  }


  namespace detail
  {
    template<std::size_t N, std::size_t i = 0, std::size_t...SDs, typename P, typename...Ds>
    constexpr auto
    derive_extents(const P& p, Ds...ds)
    {
      if constexpr (i < N)
      {
        auto d = coordinates::get_dimension(collections::get<i>(p));
        if constexpr (values::fixed<decltype(d)>)
          return derive_extents<N, i + 1, SDs..., values::fixed_value_of_v<decltype(d)>>(p, std::move(ds)...);
        else
          return derive_extents<N, i + 1, SDs..., stdex::dynamic_extent>(p, std::move(ds)..., std::move(d));
      }
      else return stdex::extents<std::size_t, SDs...>{std::move(ds)...};
    }
  }


  /**
   * \overload
   * \brief \ref Make a \ref constant_object based on a \ref coordinates::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<values::value C, coordinates::pattern_collection P> requires values::fixed<collections::size_of<P>>
  constexpr constant_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    values::value<C> and
    coordinates::pattern_collection<P> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_constant(C c, P&& p)
  {
    return attach_pattern(
      make_constant(std::move(c), detail::derive_extents<collections::size_of_v<P>>(p)),
      std::forward<P>(p));
  }


  /**
   * \overload
   * \brief The \ref coordinates::pattern_collection "pattern_collection" is constructed from a list of \ref coordinates::patterns "patterns".
   */
#ifdef __cpp_concepts
  template<values::value C, coordinates::pattern...Ps>
  constexpr constant_object auto
#else
  template<typename C, typename...Ps, std::enable_if_t<values::value<C> and (... and coordinates::pattern<Ps>), int> = 0>
  constexpr auto
#endif
  make_constant(C c, Ps&&...ps)
  {
    return make_constant(std::move(c), std::tuple{std::forward<Ps>(ps)...});
  }


  /**
   * \overload
   * \brief \ref Make a \ref constant_object based on a default-initializable \ref coordinates::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<coordinates::pattern_collection P, values::value C> requires
    std::default_initializable<P> and
    values::fixed<collections::size_of<P>>
  constexpr constant_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    coordinates::pattern_collection<P> and
    values::value<C> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_constant(C c)
  {
    return make_constant(std::move(c), P{});
  }


}

#endif
