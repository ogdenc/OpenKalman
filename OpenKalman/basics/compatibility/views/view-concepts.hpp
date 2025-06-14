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
  // view, enable_view, view_base
  // ---

  struct view_base {};

  namespace detail
  {
    template<typename T, typename U, std::enable_if_t<not std::is_same_v<T, view_interface<U>>, int> = 0>
    void is_derived_from_view_interface_test(const T&, const view_interface<U>&); // no need to define


    template<typename T, typename = void>
    struct is_derived_from_view_interface : std::false_type {};

    template<typename T>
    struct is_derived_from_view_interface<T,
      std::void_t<decltype(is_derived_from_view_interface_test(std::declval<T>(), std::declval<T>()))>> : std::true_type {};
  }


  template<class T>
  inline constexpr bool enable_view =
   (std::is_base_of_v<view_base, T> and std::is_convertible_v<const volatile T&, const volatile view_base&>) or
   detail::is_derived_from_view_interface<T>::value;


  template<typename T>
  inline constexpr bool view = range<T> and movable<T> and enable_view<T>;


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
