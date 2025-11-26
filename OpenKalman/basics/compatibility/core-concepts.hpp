/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to standard c++ library concepts.
 */

#ifndef OPENKALMAN_COMPATIBILITY_CORE_CONCEPTS_HPP
#define OPENKALMAN_COMPATIBILITY_CORE_CONCEPTS_HPP

#include "language-features.hpp"

namespace OpenKalman::stdex
{
#ifdef __cpp_lib_concepts
  using std::same_as;
  using std::derived_from;
  using std::convertible_to;
  using std::integral;
  using std::signed_integral;
  using std::unsigned_integral;
  using std::floating_point;
  using std::destructible;
  using std::constructible_from;
  using std::default_initializable;
  using std::move_constructible;
  using std::copy_constructible;
#else
  template<typename T, typename U>
  inline constexpr bool
  same_as = std::is_same_v<T, U> and std::is_same_v<U, T>;


  template<typename Derived, typename Base>
  inline constexpr bool
  derived_from =
    std::is_base_of_v<Base, Derived> and
    std::is_convertible_v<const volatile Derived*, const volatile Base*>;


  namespace detail
  {
    template<typename From, typename To, typename = void>
    struct convertible_to_impl : std::false_type {};

    template<typename From, typename To>
    struct convertible_to_impl<From, To, std::void_t<decltype(static_cast<To>(std::declval<From>()))>> : std::true_type {};
  }


  template<typename From, typename To>
  inline constexpr bool
  convertible_to = std::is_convertible_v<From, To> and detail::convertible_to_impl<From, To>::value;


  template<typename T>
  inline constexpr bool
  integral = std::is_integral_v<T>;


  template<typename T>
  inline constexpr bool
  signed_integral = integral<T> and std::is_signed_v<T>;


  template<typename T>
  inline constexpr bool
  unsigned_integral = integral<T> and not std::is_signed_v<T>;


  template<typename T>
  inline constexpr bool
  floating_point = std::is_floating_point_v<T>;


  template<typename T, typename...Args>
  inline constexpr bool
  destructible = std::is_nothrow_destructible_v<T>;


  template<typename T, typename...Args>
  inline constexpr bool
  constructible_from = destructible<T> and std::is_constructible_v<T, Args...>;


  namespace detail
  {
    template<typename T, typename = void>
    struct default_initializable_impl : std::false_type {};

    template<typename T>
    struct default_initializable_impl<T, std::void_t<decltype(T{}), decltype(::new T)>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool
  default_initializable = constructible_from<T> and detail::default_initializable_impl<T>::value;


  template<typename T>
  inline constexpr bool
  move_constructible = constructible_from<T, T> and convertible_to<T, T>;


  template<typename T>
  inline constexpr bool
  copy_constructible =
    move_constructible<T> and
    constructible_from<T, T&> and convertible_to<T&, T> and
    constructible_from<T, const T&> && convertible_to<const T&, T> and
    constructible_from<T, const T> && convertible_to<const T, T>;

#endif
}


#endif