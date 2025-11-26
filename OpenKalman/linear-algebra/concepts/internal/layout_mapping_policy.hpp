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
 * \brief Definition for \ref layout_mapping_policy.
 */

#ifndef OPENKALMAN_LAYOUT_MAPPING_POLICY_HPP
#define OPENKALMAN_LAYOUT_MAPPING_POLICY_HPP

#include "basics/basics.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename E>
    struct is_extents : std::false_type {};

    template<typename IndexType, std::size_t...Extents>
    struct is_extents<stdex::extents<IndexType, Extents...>> : std::true_type {};
  }


  /**
   * \brief MP is a LayoutMapping.
   */
  template<typename M>
#ifdef __cpp_concepts
  concept layout_mapping =
#else
  constexpr bool layout_mapping =
#endif
    stdex::copyable<M> and
    stdex::equality_comparable<M> and
    std::is_nothrow_move_constructible_v<M> and
    std::is_nothrow_move_assignable_v<M> and
    std::is_nothrow_swappable_v<M> and
    detail::is_extents<typename M::extents_type>::value and
    stdex::same_as<typename M::index_type, typename M::extents_type::index_type> and
    stdex::same_as<typename M::rank_type, typename M::extents_type::rank_type>;


  namespace detail
  {
    template<typename MP, typename M, typename E>
#ifdef __cpp_concepts
    concept layout_mapping_policy_impl_impl =
#else
    constexpr bool layout_mapping_policy_impl_impl =
#endif
      layout_mapping<M> and
      stdex::same_as<MP, typename M::layout_type> and
      stdex::same_as<E, typename M::extents_type>;


    template<typename MP, typename E>
#ifdef __cpp_concepts
    concept layout_mapping_policy_impl =
#else
    constexpr bool layout_mapping_policy_impl =
#endif
      layout_mapping_policy_impl_impl<MP, typename MP::template mapping<E>, E>;

  }


  /**
   * \brief MP is a LayoutMappingPolicy.
   */
  template<typename MP>
#ifdef __cpp_concepts
  concept layout_mapping_policy =
#else
  constexpr bool layout_mapping_policy =
#endif
    detail::layout_mapping_policy_impl<MP, stdex::extents<std::size_t>>;

}

#endif
