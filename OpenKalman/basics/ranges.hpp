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
 * \brief Definitions implementing features of the c++ ranges library for compatibility.
 */

#ifndef OPENKALMAN_RANGES_HPP
#define OPENKALMAN_RANGES_HPP

#ifdef __cpp_lib_ranges
namespace OpenKalman::ranges
{
  using namespace std::ranges;
}
#else

#include <iterator>
#include "language-features.hpp"

namespace OpenKalman
{
  template<typename T>
  using iter_value_t = typename std::iterator_traits<T>::value_type;

  template<typename T>
  using iter_reference_t = decltype(*std::declval<T&>());

  template<typename T>
  using iter_difference_t = typename std::iterator_traits<std::decay_t<T>>::difference_type;
}


namespace OpenKalman::ranges
{
  // ---
  // enable_borrowed_range
  // ---

  template<typename T>
  inline constexpr bool enable_borrowed_range = false;


  // ---
  // begin, cbegin
  // ---

  namespace detail_begin
  {
    using namespace std;

    template<typename T, typename = void> struct begin_def : std::false_type {};
    template<typename T> struct begin_def<T, std::void_t<decltype(begin(std::declval<T&&>()))>> : std::true_type {};

    struct begin_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (std::is_array_v<std::remove_reference_t<T>> or begin_def<T>::value), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (std::is_array_v<std::remove_reference_t<T>>)
          return t + 0;
        else
          return internal::decay_copy(begin(std::forward<T>(t)));
      }
    };

    template<typename T>
    using CT = std::conditional_t<std::is_lvalue_reference_v<T>, const std::remove_reference_t<T>&, const T>;

    struct cbegin_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (std::is_array_v<std::remove_reference_t<T>> or begin_def<CT<T>>::value), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        return begin_impl{}(static_cast<CT<T>&&>(t));
      }
    };

  }

  inline constexpr detail_begin::begin_impl begin;
  inline constexpr detail_begin::cbegin_impl cbegin;


  // ---
  // end, cend
  // ---

  namespace detail_end
  {
    using namespace std;

    template<typename T, typename = void> struct end_def : std::false_type {};
    template<typename T> struct end_def<T, std::void_t<decltype(end(std::declval<T&&>()))>> : std::true_type {};

    struct end_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (is_bounded_array_v<std::remove_reference_t<T>> or end_def<T>::value), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (is_bounded_array_v<remove_cvref_t<T>>)
          return t + std::extent_v<remove_cvref_t<T>>;
        else
          return internal::decay_copy(end(std::forward<T>(t)));
      }
    };

    template<typename T>
    using CT = std::conditional_t<std::is_lvalue_reference_v<T>, const std::remove_reference_t<T>&, const T>;

    struct cend_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (is_bounded_array_v<std::remove_reference_t<T>> or end_def<CT<T>>::value), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        return end_impl{}(static_cast<CT<T>&&>(t));
      }
    };
  }

  inline constexpr detail_end::end_impl end;
  inline constexpr detail_end::cend_impl cend;


  // ---
  // rbegin, crbegin
  // ---

  namespace detail_rbegin
  {
    template<typename T, typename = void> struct rbegin_def : std::false_type {};
    template<typename T> struct rbegin_def<T, std::void_t<decltype(rbegin(std::declval<T&&>()))>> : std::true_type {};

    struct rbegin_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (rbegin_def<T>::value or (detail_begin::begin_def<T>::value and detail_end::end_def<T>::value)), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (rbegin_def<T>::value)
          return internal::decay_copy(rbegin(std::forward<T>(t)));
        else
          return std::make_reverse_iterator(end(std::forward<T>(t)));
      }
    };

    template<typename T>
    using CT = std::conditional_t<std::is_lvalue_reference_v<T>, const std::remove_reference_t<T>&, const T>;

    struct crbegin_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (rbegin_def<T>::value or (detail_begin::begin_def<T>::value and detail_end::end_def<T>::value)), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        return rbegin_impl{}(static_cast<CT<T>&&>(t));
      }
    };

  }

  inline constexpr detail_rbegin::rbegin_impl rbegin;
  inline constexpr detail_rbegin::crbegin_impl crbegin;


  // ---
  // rend, crend
  // ---

  namespace detail_rend
  {
    using namespace std;

    template<typename T, typename = void> struct rend_def : std::false_type {};
    template<typename T> struct rend_def<T, std::void_t<decltype(rend(std::declval<T&&>()))>> : std::true_type {};

    struct rend_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (rend_def<T>::value or (detail_begin::begin_def<T>::value and detail_end::end_def<T>::value)), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (rend_def<T>::value)
          return internal::decay_copy(rend(std::forward<T>(t)));
        else
          return std::make_reverse_iterator(begin(std::forward<T>(t)));
      }
    };

    template<typename T>
    using CT = std::conditional_t<std::is_lvalue_reference_v<T>, const std::remove_reference_t<T>&, const T>;

    struct crend_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (rend_def<T>::value or (detail_begin::begin_def<T>::value and detail_end::end_def<T>::value)), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        return rend_impl{}(static_cast<CT<T>&&>(t));
      }
    };
  }

  inline constexpr detail_rend::rend_impl rend;
  inline constexpr detail_rend::crend_impl crend;


  // ---
  // size
  // ---

  namespace detail_size
  {
    using namespace std;

    struct size_impl
    {
    private:

      template<typename T, typename = void> struct member_size_def : std::false_type {};
      template<typename T> struct member_size_def<T, std::void_t<decltype(std::declval<T&&>().size())>> : std::true_type {};

      template<typename T, typename = void> struct atd_size_def : std::false_type {};
      template<typename T> struct atd_size_def<T, std::void_t<decltype(size(std::declval<T&&>()))>> : std::true_type {};

    public:

      template<typename T, std::enable_if_t<is_bounded_array_v<std::remove_reference_t<T>> or
        member_size_def<T>::value or atd_size_def<T>::value, int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (is_bounded_array_v<remove_cvref_t<T>>)
          return internal::decay_copy(std::extent_v<remove_cvref_t<T>>);
        else if constexpr (member_size_def<T>::value)
          return internal::decay_copy(t.size());
        else if constexpr (atd_size_def<T>::value)
          return internal::decay_copy(size(t));
        else
          return end(t) - begin(t);
      }
    };
  }

  inline constexpr detail_size::size_impl size;


  // ---
  // iterator_t, const_iterator_t, sentinel_t, const_sentinel_t
  // ---

  template<typename R>
  using iterator_t = decltype(begin(std::declval<R&>()));

  template<typename R>
  using const_iterator_t = decltype(cbegin(std::declval<R&>()));


  template<typename R>
  using sentinel_t = decltype(end(std::declval<R&>()));

  template<typename R>
  using const_sentinel_t = decltype(cend(std::declval<R&>()));


  // ---
  // range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct is_range : std::false_type {};

    template<typename T>
    struct is_range<T, std::void_t<iterator_t<T>, sentinel_t<T>>> : std::true_type {};
  }

  template<typename T>
  constexpr bool range = detail::is_range<T>::value;


  // ---
  // random_access_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct is_random_access_range : std::false_type {};

    template<typename T>
    struct is_random_access_range<T, std::enable_if_t<
      std::is_same_v<decltype(*std::declval<iterator_t<T>>()), iter_reference_t<iterator_t<T>>> and
      std::is_same_v<decltype(std::declval<iterator_t<T>>() + std::declval<iter_difference_t<iterator_t<T>>>()), iterator_t<T>> and
      std::is_same_v<decltype(std::declval<iter_difference_t<iterator_t<T>>>() + std::declval<iterator_t<T>>()), iterator_t<T>> and
      std::is_same_v<decltype(std::declval<iterator_t<T>>() - std::declval<iter_difference_t<iterator_t<T>>>()), iterator_t<T>>
      >>
    : std::true_type {};
  }

  template<typename T>
  constexpr bool random_access_range = range<T> and detail::is_random_access_range<T>::value;


  // ---
  // sized_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct is_sized_range : std::false_type {};

    template<typename T>
    struct is_sized_range<T, std::void_t<decltype(size(std::declval<T&>()))>> : std::true_type {};

  }

  template<typename T>
  constexpr bool sized_range = range<T> and detail::is_sized_range<T>::value;


  // ---
  // borrowed_range
  // ---

  namespace detail_borrowed_range
  {
#ifdef __cpp_lib_remove_cvref
    using std::remove_cvref_t;
#endif


    template<typename R, typename = void>
    struct is_borrowed_range : std::false_type {};

    template<typename R>
    struct is_borrowed_range<R, std::enable_if_t<range<R> and
      (std::is_lvalue_reference_v<R> or enable_borrowed_range<remove_cvref_t<R>>)>> : std::true_type {};
  }

  template<typename T>
  constexpr bool borrowed_range = detail_borrowed_range::is_borrowed_range<T>::value;


  // ---
  // range_size_t, range_difference_t, range_value_t
  // ---

  template<typename R, std::enable_if_t<sized_range<R>, int> = 0>
  using range_size_t = decltype(size(std::declval<R&>()));


  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_difference_t = iter_difference_t<iterator_t<R>>;


  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_value_t = iter_value_t<iterator_t<R>>;


  // ---
  // view_interface
  // ---

  template<typename Derived>
  struct view_interface
  {
    template<typename R = Derived, std::enable_if_t<sized_range<R>, int> = 0>
    [[nodiscard]] constexpr bool
    empty() { return ranges::size(static_cast<R&>(*this)) == 0; }

    template<typename R = Derived, std::enable_if_t<sized_range<R>, int> = 0>
    [[nodiscard]] constexpr bool
    empty() const { return ranges::size(static_cast<const R&>(*this)) == 0; }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr auto
    cbegin() { return ranges::cbegin(static_cast<R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr auto
    cbegin() const { return ranges::cbegin(static_cast<const R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr auto
    cend() { return ranges::cend(static_cast<R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr auto
    cend() const { return ranges::cend(static_cast<const R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<sized_range<R>, int> = 0>
    constexpr explicit
    operator bool() { return not empty(); }

    template<typename R = Derived, std::enable_if_t<sized_range<R>, int> = 0>
    constexpr explicit
    operator bool() const { return not empty(); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0,
      typename = std::void_t<decltype(end(std::declval<R&>()) - begin(std::declval<R&>()))>>
    constexpr auto
    size() { return end(static_cast<R&>(*this)) - begin(static_cast<R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0,
      typename = std::void_t<decltype(end(std::declval<R&>()) - begin(std::declval<R&>()))>>
    constexpr auto
    size() const { return end(static_cast<const R&>(*this)) - begin(static_cast<const R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr decltype(auto)
    front() { return *begin(static_cast<R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr decltype(auto)
    front() const { return *begin(static_cast<const R&>(*this)); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr decltype(auto)
    back() { return *std::prev(end(static_cast<R&>(*this))); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr decltype(auto)
    back() const { return *std::prev(end(static_cast<const R&>(*this))); }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr decltype(auto)
    operator[](range_difference_t<R> n) { return begin(static_cast<R&>(*this))[n]; }

    template<typename R = Derived, std::enable_if_t<range<R>, int> = 0>
    constexpr decltype(auto)
    operator[](range_difference_t<R> n) const { return begin(static_cast<const R&>(*this))[n]; }
  };


  // ---
  // view
  // ---

  template<typename T>
  constexpr bool view = range<T> and std::is_object_v<T> and
    std::is_move_constructible_v<T> and std::is_assignable_v<T&, T> and
    std::is_base_of_v<view_interface<remove_cvref_t<T>>, remove_cvref_t<T>>;


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
        (std::is_object_v<std::remove_reference_t<T>> and std::is_move_constructible_v<std::remove_reference_t<T>> and
          std::is_assignable_v<std::remove_reference_t<T>&, std::remove_reference_t<T>> and
          not detail::is_initializer_list<remove_cvref_t<T>>::value))));


} // namespace OpenKalman::ranges

#endif


#endif //OPENKALMAN_RANGES_HPP
