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
    struct is_extents<stdcompat::extents<IndexType, Extents...>> : std::true_type {};
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
    stdcompat::copyable<typename MP::mapping> and
    stdcompat::equality_comparable<typename MP::mapping> and
    std::is_nothrow_move_constructible_v<typename MP::mapping> and
    std::is_nothrow_move_assignable_v<typename MP::mapping> and
    std::is_nothrow_swappable_v<typename MP::mapping> and
    stdcompat::same_as<typename MP::mapping::layout_type, MP> and
    detail::is_extents<typename MP::mapping::extents_type>::value;

}

#endif
