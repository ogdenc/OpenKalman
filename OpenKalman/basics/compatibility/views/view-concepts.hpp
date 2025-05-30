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
 * \internal
 * \brief Definitions of concepts equivalent to STL view-related concepts, for compatibility.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_VIEW_CONCEPTS_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_VIEW_CONCEPTS_HPP

#ifndef __cpp_lib_ranges

#include "basics/global-definitions.hpp"
#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/ranges.hpp"
#include "view_interface.hpp"

namespace OpenKalman::ranges
{
  // ---
  // view
  // ---

  template<typename T>
  inline constexpr bool view = range<T> and movable<T> and std::is_base_of_v<view_interface<T>, T>;


  // ---
  // viewable_range
  // ---

  namespace detail
  {
   template<typename T>
   struct is_initializer_list : std::false_type {};

   template<typename T>
   struct is_initializer_list<std::initializer_list<T>> : std::true_type {};
  }


  template<typename T>
  constexpr bool viewable_range =  ranges::range<T> and
    ((view<remove_cvref_t<T>> and std::is_constructible_v<remove_cvref_t<T>, T>) or
     (not view<remove_cvref_t<T>> and
      (std::is_lvalue_reference_v<T> or
        (movable<std::remove_reference_t<T>> and not OpenKalman::internal::is_initializer_list<remove_cvref_t<T>>::value))));

}


#endif

#endif //OPENKALMAN_COMPATIBILITY_VIEWS_VIEW_CONCEPTS_HPP
