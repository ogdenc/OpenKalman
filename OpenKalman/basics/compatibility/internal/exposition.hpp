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
 * \brief Exposition-only definitions from teh c++ language standard.
 */

#ifndef OPENKALMAN_STD_EXPOSITION_FEATURES_HPP
#define OPENKALMAN_STD_EXPOSITION_FEATURES_HPP

#include <type_traits>
#include <utility>
#include "basics/compatibility/core-concepts.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    struct decay_copy_impl final
    {
      template<typename T>
      constexpr std::decay_t<T> operator()(T&& t) const noexcept { return std::forward<T>(t); }
    };
  }

  inline constexpr detail::decay_copy_impl decay_copy;


  namespace detail
  {
    template<typename T, typename = void>
    struct is_integer_like_impl : std::false_type {};

    template<>
    struct is_integer_like_impl<bool> : std::false_type {};

    template<>
    struct is_integer_like_impl<const bool> : std::false_type {};

    template<>
    struct is_integer_like_impl<volatile bool> : std::false_type {};

    template<>
    struct is_integer_like_impl<const volatile bool> : std::false_type {};

    template<typename T>
    struct is_integer_like_impl<T, std::enable_if_t<std::is_integral_v<T>>> : std::true_type {};
  }

  template<typename T>
  inline constexpr bool is_integer_like = detail::is_integer_like_impl<T>::value;

  template<typename T>
  inline constexpr bool is_signed_integer_like = is_integer_like<T> and std::is_signed_v<T>;

  template<typename T>
  inline constexpr bool is_unsigned_integer_like = is_integer_like<T> and std::is_unsigned_v<T>;


  // -------------------- //
  //  is_initializer_list //
  // -------------------- //

  /**
   * \brief Whether the argument is a specialization of std::initializer_list
   */
  template<typename T>
  struct is_initializer_list : std::false_type {};

  /// \overload
  template<typename T>
  struct is_initializer_list<std::initializer_list<T>> : std::true_type {};

  /// \overload
  template<typename T>
  struct is_initializer_list<T&> : is_initializer_list<T> {};

  /// \overload
  template<typename T>
  struct is_initializer_list<T&&> : is_initializer_list<T> {};

  /// \overload
  template<typename T>
  struct is_initializer_list<const T> : is_initializer_list<T> {};

  /// \overload
  template<typename T>
  struct is_initializer_list<volatile T> : is_initializer_list<T> {};


  // ----------------- //
  //  boolean_testable //
  // ----------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename B>
    concept boolean_testable_impl = std::convertible_to<B, bool>;
#else
    template<typename B, typename = void>
    struct boolean_testable_impl1 : std::false_type {};

    template<typename B>
    struct boolean_testable_impl1<B, std::enable_if_t<stdex::convertible_to<B, bool>>> : std::true_type {};


    template<typename B, typename = void>
    struct boolean_testable_impl2 : std::false_type {};

    template<typename B>
    struct boolean_testable_impl2<B, std::enable_if_t<
      detail::boolean_testable_impl1<B>::value and detail::boolean_testable_impl1<decltype(not std::declval<B>())>::value>> : std::true_type {};
#endif
  }


#ifdef __cpp_concepts
  template<typename B>
  concept boolean_testable =
    detail::boolean_testable_impl<B> and
    requires (B&& b) { { not std::forward<B>(b) } -> detail::boolean_testable_impl; };
#else
  template<typename B>
  inline constexpr bool boolean_testable =
    detail::boolean_testable_impl1<B>::value and detail::boolean_testable_impl2<B>::value;
#endif


  // ----------------------------- //
  //  WeaklyEqualityComparableWith //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<typename T, typename U>
  concept WeaklyEqualityComparableWith =
    requires(const std::remove_reference_t<T>& t, const std::remove_reference_t<U>& u) {
      { t == u } -> boolean_testable;
      { t != u } -> boolean_testable;
      { u == t } -> boolean_testable;
      { u != t } -> boolean_testable;
    };
#else
  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct WeaklyEqualityComparableWithImpl : std::false_type {};

    template<typename T, typename U>
    struct WeaklyEqualityComparableWithImpl<T, U, std::enable_if_t<
      boolean_testable<decltype(std::declval<const std::remove_reference_t<T>&>() == std::declval<const std::remove_reference_t<U>&>())> and
      boolean_testable<decltype(std::declval<const std::remove_reference_t<T>&>() != std::declval<const std::remove_reference_t<U>&>())> and
      boolean_testable<decltype(std::declval<const std::remove_reference_t<U>&>() == std::declval<const std::remove_reference_t<T>&>())> and
      boolean_testable<decltype(std::declval<const std::remove_reference_t<U>&>() != std::declval<const std::remove_reference_t<T>&>())>
      >> : std::true_type {};
  }

  template<typename T, typename U>
  inline constexpr bool
  WeaklyEqualityComparableWith = detail::WeaklyEqualityComparableWithImpl<T, U>::value;
#endif


  // --------------------- //
  //  PartiallyOrderedWith //
  // --------------------- //

#ifdef __cpp_concepts
  template<typename T, typename U>
  concept PartiallyOrderedWith =
    requires(const std::remove_reference_t<T>& t, const std::remove_reference_t<U>& u) {
      { t <  u } -> boolean_testable;
      { t >  u } -> boolean_testable;
      { t <= u } -> boolean_testable;
      { t >= u } -> boolean_testable;
      { u <  t } -> boolean_testable;
      { u >  t } -> boolean_testable;
      { u <= t } -> boolean_testable;
      { u >= t } -> boolean_testable;
    };
#else
  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct PartiallyOrderedWithImpl : std::false_type {};

    template<typename T, typename U>
    struct PartiallyOrderedWithImpl<T, U, std::enable_if_t<
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<T>&>() < std::declval<const std::remove_reference_t<U>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<T>&>() > std::declval<const std::remove_reference_t<U>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<T>&>() <= std::declval<const std::remove_reference_t<U>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<T>&>() >= std::declval<const std::remove_reference_t<U>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<U>&>() < std::declval<const std::remove_reference_t<T>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<U>&>() > std::declval<const std::remove_reference_t<T>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<U>&>() <= std::declval<const std::remove_reference_t<T>&>())> and
      OpenKalman::internal::boolean_testable<decltype(std::declval<const std::remove_reference_t<U>&>() >= std::declval<const std::remove_reference_t<T>&>())>
      >> : std::true_type {};
  }

  template<typename T, typename U>
  inline constexpr bool
  PartiallyOrderedWith = detail::PartiallyOrderedWithImpl<T, U>::value;
#endif


}


#endif