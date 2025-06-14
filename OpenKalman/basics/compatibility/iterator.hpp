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
 * \brief Header file for compatibility definitions equivalent to those in the STL iterator library.
 */

#ifndef OPENKALMAN_COMPATIBILITY_ITERATOR_HPP
#define OPENKALMAN_COMPATIBILITY_ITERATOR_HPP

#ifndef __cpp_lib_ranges

#include <iterator>
#include "language-features.hpp"
#include "common_reference.hpp"

namespace OpenKalman
{
  // ---
  // indirectly_readable_traits
  // ---

  namespace detail_indirectly_readable
  {
    struct no_value_type {};

    template<typename, typename = void>
    struct cond_value_type {};

    template<typename T>
    struct cond_value_type<T, std::enable_if_t<std::is_object_v<T>>> { using value_type = std::remove_cv_t<T>; };

    template<typename, typename = void>
    struct has_member_value_type_impl : std::false_type {};

    template<typename T>
    struct has_member_value_type_impl<T, std::void_t<typename T::value_type>> : std::true_type {};

    template<typename T>
    inline constexpr bool has_member_value_type = has_member_value_type_impl<T>::value;

    template<typename, typename = void>
    struct has_member_element_type_impl : std::false_type {};

    template<typename T>
    struct has_member_element_type_impl<T, std::void_t<typename T::element_type>> : std::true_type {};

    template<typename T>
    inline constexpr bool has_member_element_type = has_member_element_type_impl<T>::value;
  }


  template<typename I, typename = void>
  struct indirectly_readable_traits {};

  template<typename T>
  struct indirectly_readable_traits<T*> : detail_indirectly_readable::cond_value_type<T> {};

  template<typename I>
  struct indirectly_readable_traits<I, std::enable_if_t<std::is_array_v<I>>>
  { using value_type = std::remove_cv_t<std::remove_extent_t<I>>; };

  template<typename T>
  struct indirectly_readable_traits<const T> : indirectly_readable_traits<T> {};

  template<typename T>
  struct indirectly_readable_traits<T, std::enable_if_t<
    detail_indirectly_readable::has_member_value_type<T> and
    not detail_indirectly_readable::has_member_element_type<T>>>
    : detail_indirectly_readable::cond_value_type<typename T::value_type> {};

  template<typename T>
  struct indirectly_readable_traits<T, std::enable_if_t<
    not detail_indirectly_readable::has_member_value_type<T> and
    detail_indirectly_readable::has_member_element_type<T>>>
    : detail_indirectly_readable::cond_value_type<typename T::element_type> {};

  template<typename T>
  struct indirectly_readable_traits<T, std::enable_if_t<
    detail_indirectly_readable::has_member_value_type<T> and
    detail_indirectly_readable::has_member_element_type<T>>>
    : std::conditional_t<
        std::is_same_v<std::remove_cv_t<typename T::element_type>, std::remove_cv_t<typename T::value_type>>,
        detail_indirectly_readable::cond_value_type<typename T::value_type>,
        detail_indirectly_readable::no_value_type> {};


  // ---
  // incrementable_traits
  // ---

  namespace detail
  {
    template<typename, typename = void>
    struct has_member_difference_type : std::false_type {};

    template<typename T>
    struct has_member_difference_type<T, std::void_t<typename T::difference_type>> : std::true_type {};
  }


  template<typename I, typename = void>
  struct incrementable_traits {};

  template<typename T>
  struct incrementable_traits<T*, std::enable_if_t<std::is_object_v<T>>>
  {
    using difference_type = std::ptrdiff_t;
  };

  template<typename T>
  struct incrementable_traits<const T> : incrementable_traits<T> {};

  template<typename T>
  struct incrementable_traits<T, std::enable_if_t<detail::has_member_difference_type<T>::value>>
  {
    using difference_type = typename T::difference_type;
  };

  template<typename T>
  struct incrementable_traits<T, std::enable_if_t<not detail::has_member_difference_type<T>::value and
    std::is_integral_v<decltype(std::declval<const T&>() - std::declval<const T&>())>>>
  {
    using difference_type = std::make_signed_t<decltype(std::declval<T>() - std::declval<T>())>;
  };


  // ---
  // primitives
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct iter_value_impl : indirectly_readable_traits<T> {};

    template<typename T>
    struct iter_value_impl<T, std::void_t<typename std::iterator_traits<T>::value_type>> : std::iterator_traits<T> {};


    template<typename T, typename = void>
    struct iter_difference_impl : incrementable_traits<T> {};

    template<typename T>
    struct iter_difference_impl<T, std::void_t<typename std::iterator_traits<T>::difference_type>> : std::iterator_traits<T> {};
  }


  template<typename I>
  using iter_value_t = typename detail::iter_value_impl<remove_cvref_t<I>>::value_type;

  template<typename I>
  using iter_reference_t = decltype(*std::declval<I&>());

  template<typename I>
  using iter_difference_t = typename detail::iter_difference_impl<remove_cvref_t<I>>::difference_type;

  template<typename I>
  using iter_rvalue_reference_t = decltype(std::move(*std::declval<I&>()));


  // ---
  // indirectly_readable
  // ---

  namespace detail
  {
    template<typename I, typename = void, typename = void>
    struct is_indirectly_readable : std::false_type {};

    template<typename I>
    struct is_indirectly_readable<I,
      std::void_t<iter_value_t<I>, iter_reference_t<I>, iter_rvalue_reference_t<I>>,
      std::enable_if_t<std::is_same<decltype(*std::declval<I>()), iter_reference_t<I>>::value>> : std::true_type {};
  }


  template<typename I>
  inline constexpr bool indirectly_readable = detail::is_indirectly_readable<remove_cvref_t<I>>::value;


  template<typename T, std::enable_if_t<indirectly_readable<T>, int > = 0>
  using iter_const_reference_t = common_reference_t<const iter_value_t<T>&&, iter_reference_t<T>>;

  template<typename T, std::enable_if_t<indirectly_readable<T>, int > = 0>
  using iter_common_reference_t = common_reference_t<iter_reference_t<T>, iter_value_t<T>&>;


  // ---
  // indirectly_writable
  // ---

  namespace detail
  {
    template<typename Out, typename T, typename = void>
    struct is_indirectly_writable : std::false_type {};

    template<typename Out, typename T>
    struct is_indirectly_writable<Out, T,
      std::void_t<
        decltype(*std::declval<Out&>() = std::declval<T&&>()),
        decltype(*std::declval<Out&&>() = std::declval<T&&>()),
        decltype(const_cast<const iter_reference_t<Out>&&>(*std::declval<Out&>()) = std::declval<T&&>()),
        decltype(const_cast<const iter_reference_t<Out>&&>(*std::declval<Out&&>()) = std::declval<T&&>())
      >> : std::true_type {};
  }


  template<typename Out, typename T>
  inline constexpr bool indirectly_writable = detail::is_indirectly_readable<Out, T>::value;


  // ---
  // weakly_incrementable
  // ---

  namespace detail
  {
    template<typename I, typename = void, typename = void>
    struct is_weakly_incrementable : std::false_type {};

    template<typename I>
    struct is_weakly_incrementable<I, std::void_t<iter_difference_t<I>, decltype(std::declval<I&>()++)>,
      std::enable_if_t<std::is_same<decltype(++std::declval<I&>()), I&>::value>> : std::true_type {};
  }


  template<typename I>
  inline constexpr bool weakly_incrementable = detail::is_weakly_incrementable<I>::value;


  // ---
  // input_or_output_iterator
  // ---

  namespace detail
  {
    template<typename I, typename = void, typename = void>
    struct is_input_or_output_iterator : std::false_type {};

    template<typename I>
    struct is_input_or_output_iterator<I,
      std::void_t<iter_value_t<I>, iter_reference_t<I>, iter_rvalue_reference_t<I>>,
      std::enable_if_t<std::is_same<decltype(*std::declval<I>()), iter_reference_t<I>>::value>> : std::true_type {};
  }


  template<typename I>
  inline constexpr bool input_or_output_iterator = detail::is_input_or_output_iterator<I>::value and weakly_incrementable<I>;


  // ---
  // input_iterator
  // ---

  template<typename I>
  inline constexpr bool input_iterator =
    input_or_output_iterator<I> and
    indirectly_readable<I>;


  // ---
  // output_iterator
  // ---

  namespace detail
  {
    template<typename I, typename T, typename = void>
    struct output_iterator_impl : std::false_type {};

    template<typename I, typename T>
    struct output_iterator_impl<I, T, std::void_t<decltype(*std::declval<I&>()++ = std::declval<T&&>())>> : std::true_type {};
  }


  template<typename I, typename T>
  inline constexpr bool output_iterator =
    input_or_output_iterator<I> and
    indirectly_writable<I, T> and
    detail::output_iterator_impl<I, T>::value;


  // ---
  // incrementable
  // ---

  namespace detail
  {
    template<typename I, typename = void>
    struct is_incrementable : std::false_type {};

    template<typename I>
    struct is_incrementable<I, std::enable_if_t<std::is_same<decltype(std::declval<I&>()++), I>::value>> : std::true_type {};
  }


  template<typename I>
  inline constexpr bool incrementable =
    std::is_copy_constructible_v<I> and
    std::is_object_v<I> and
    std::is_move_constructible_v<I> and
    std::is_assignable_v<I&, I> and
    std::is_assignable_v<I&, I&> and
    std::is_assignable_v<I&, const I&> and
    std::is_assignable_v<I&, const I> and
    std::is_default_constructible_v<I> and
    weakly_incrementable<I> and
    detail::is_incrementable<I>::value;


  // ---
  // forward_iterator
  // ---

  template<typename I>
  inline constexpr bool forward_iterator =
    input_iterator<I> and
    incrementable<I>;


  // ---
  // bidirectional_iterator
  // ---

  namespace detail
  {
    template<typename I, typename = void>
    struct is_bidirectional_iterator : std::false_type {};

    template<typename I>
    struct is_bidirectional_iterator<I, std::enable_if_t<
      std::is_same<decltype(--std::declval<I&>()), I&>::value and
      std::is_same<decltype(std::declval<I&>()--), I>::value>> : std::true_type {};
  }


  template<typename I>
  inline constexpr bool bidirectional_iterator =
    forward_iterator<I> and
    detail::is_bidirectional_iterator<I>::value;


  // ---
  // random_access_iterator
  // ---

  namespace detail
  {
    template<typename I, typename = void>
    struct is_random_access_iterator : std::false_type {};

    template<typename I>
    struct is_random_access_iterator<I, std::enable_if_t<
      std::is_same_v<decltype(std::declval<I&>() += std::declval<iter_difference_t<I>>()), I&> and
      std::is_same_v<decltype(std::declval<const I&>() + std::declval<iter_difference_t<I>>()), I> and
      std::is_same_v<decltype(std::declval<iter_difference_t<I>>() + std::declval<const I&>()), I> and
      std::is_same_v<decltype(std::declval<I&>() -= std::declval<iter_difference_t<I>>()), I&> and
      std::is_same_v<decltype(std::declval<const I&>() - std::declval<iter_difference_t<I>>()), I> and
      std::is_same_v<decltype(std::declval<const I&>()[std::declval<iter_difference_t<I>>()]), iter_reference_t<I>>
    >> : std::true_type {};
  }


  template<typename I>
  inline constexpr bool random_access_iterator =
    bidirectional_iterator<I> and
    detail::is_random_access_iterator<I>::value;


  // ---
  // sentinel_for, sized_sentinel_for
  // ---

  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct WeaklyEqualityComparableWith : std::false_type {};

    template<typename T, typename U>
    struct WeaklyEqualityComparableWith<T, U, std::enable_if_t<
      std::is_convertible_v<decltype(std::declval<const T&>() == std::declval<const U&>()), bool> and
      std::is_convertible_v<decltype(std::declval<const T&>() != std::declval<const U&>()), bool> and
      std::is_convertible_v<decltype(std::declval<const U&>() == std::declval<const T&>()), bool> and
      std::is_convertible_v<decltype(std::declval<const U&>() != std::declval<const T&>()), bool>
    >> : std::true_type {};

    template<typename T, typename U, typename = void>
    struct subtractable : std::false_type {};

    template<typename I, typename S>
    struct subtractable<I, S, std::enable_if_t<
      std::is_same_v<decltype(std::declval<const S&>() - std::declval<const I&>()), iter_difference_t<I>> and
      std::is_same_v<decltype(std::declval<const I&>() - std::declval<const S&>()), iter_difference_t<I>>
    >> : std::true_type {};

  }


  template<typename S, typename I>
  inline constexpr bool sentinel_for =
    semiregular<S> and input_or_output_iterator<I> and detail::WeaklyEqualityComparableWith<S, I>::value;


  template<typename S, typename I>
  inline constexpr bool sized_sentinel_for = sentinel_for<S, I> and
    /*not std::disable_sized_sentinel_for<std::remove_cv_t<S>, std::remove_cv_t<I>> and */
    detail::subtractable<I, S>::value;


  // ---
  // unreachable_sentinel_t, unreachable_sentinel
  // ---

  struct unreachable_sentinel_t
  {
    template<typename I, std::enable_if_t<weakly_incrementable<I>, int> = 0>
    friend constexpr bool
    operator==(unreachable_sentinel_t, unreachable_sentinel_t) noexcept { return false; };

    template<typename I, std::enable_if_t<weakly_incrementable<I>, int> = 0>
    friend constexpr bool
    operator==(unreachable_sentinel_t, const I&) noexcept { return false; };

    template<typename I, std::enable_if_t<weakly_incrementable<I>, int> = 0>
    friend constexpr bool
    operator==(const I&, unreachable_sentinel_t) noexcept { return false; };
  };


  inline constexpr unreachable_sentinel_t unreachable_sentinel {};


}

#endif

#endif //OPENKALMAN_COMPATIBILITY_ITERATOR_HPP
