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
 * \internal
 * \file
 * \brief Definitions equivalent to std::ranges customization-point objects for compatibility.
 */

#ifndef OPENKALMAN_COMPATIBILITY_RANGES_RANGE_ACCESS_HPP
#define OPENKALMAN_COMPATIBILITY_RANGES_RANGE_ACCESS_HPP

#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/internal/exposition.hpp"
#include "basics/compatibility/iterator.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::enable_borrowed_range;
  using std::ranges::begin;
  using std::ranges::cbegin;
  using std::ranges::end;
  using std::ranges::cend;
  using std::ranges::rbegin;
  using std::ranges::crbegin;
  using std::ranges::rend;
  using std::ranges::crend;
  using std::ranges::size;
  using std::ranges::empty;
  using std::ranges::advance;
  using std::ranges::next;
  using std::ranges::iterator_t;
  using std::ranges::const_iterator_t;
  using std::ranges::sentinel_t;
  using std::ranges::const_sentinel_t;
#else
  // Forward definition
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
          return OpenKalman::internal::decay_copy(begin(std::forward<T>(t)));
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
        (stdcompat::is_bounded_array_v<std::remove_reference_t<T>> or end_def<T>::value), int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (stdcompat::is_bounded_array_v<stdcompat::remove_cvref_t<T>>)
          return t + std::extent_v<stdcompat::remove_cvref_t<T>>;
        else
          return OpenKalman::internal::decay_copy(end(std::forward<T>(t)));
      }
    };

    template<typename T>
    using CT = std::conditional_t<std::is_lvalue_reference_v<T>, const std::remove_reference_t<T>&, const T>;

    struct cend_impl
    {
      template<typename T, std::enable_if_t<(std::is_lvalue_reference_v<T> or enable_borrowed_range<std::remove_cv_t<T>>) and
        (stdcompat::is_bounded_array_v<std::remove_reference_t<T>> or end_def<CT<T>>::value), int> = 0>
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
          return OpenKalman::internal::decay_copy(rbegin(std::forward<T>(t)));
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
          return OpenKalman::internal::decay_copy(rend(std::forward<T>(t)));
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

      template<typename T, std::enable_if_t<stdcompat::is_bounded_array_v<std::remove_reference_t<T>> or
        member_size_def<T>::value or atd_size_def<T>::value, int> = 0>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (stdcompat::is_bounded_array_v<stdcompat::remove_cvref_t<T>>)
          return OpenKalman::internal::decay_copy(std::extent_v<stdcompat::remove_cvref_t<T>>);
        else if constexpr (member_size_def<T>::value)
          return OpenKalman::internal::decay_copy(std::forward<T>(t).size());
        else if constexpr (atd_size_def<T>::value)
          return OpenKalman::internal::decay_copy(size(std::forward<T>(t)));
        else
          return end(std::forward<T>(t)) - begin(std::forward<T>(t));
      }
    };
  }

  inline constexpr detail_size::size_impl size;


  // ---
  // empty
  // ---

  namespace detail_empty
  {
    using namespace std;

    struct empty_impl
    {
    private:

      template<typename T, typename = void> struct member_empty_def : std::false_type {};
      template<typename T> struct member_empty_def<T, std::void_t<decltype(bool(std::declval<T&&>().empty()))>> : std::true_type {};

      template<typename T, typename = void> struct ranges_size_compares_zero : std::false_type {};
      template<typename T> struct ranges_size_compares_zero<T, std::void_t<decltype(stdcompat::ranges::size(std::declval<T&&>()) == 0)>> : std::true_type {};

      template<typename T, typename = void> struct begin_at_end : std::false_type {};
      template<typename T> struct begin_at_end<T, std::void_t<decltype(bool(stdcompat::ranges::begin(std::declval<T&&>()) == stdcompat::ranges::end(std::declval<T&&>())))>> : std::true_type {};

    public:

      template<typename T, std::enable_if_t<member_empty_def<T>::value or ranges_size_compares_zero<T>::value or begin_at_end<T>::value, int> = 0>
      constexpr bool
      operator() [[nodiscard]] (T&& t) const
      {
        if constexpr (member_empty_def<T>::value)
          return bool(std::forward<T>(t).empty());
        else if constexpr (ranges_size_compares_zero<T>::value)
          return stdcompat::ranges::size(std::forward<T>(t)) == 0;
        else
          return bool(stdcompat::ranges::begin(std::forward<T>(t)) == stdcompat::ranges::end(std::forward<T>(t)));
      }
    };
  }

  inline constexpr detail_empty::empty_impl empty;


  // ---
  // advance, next
  // ---

  namespace detail_advance
  {
    struct advance_fn
    {
      template<typename I, std::enable_if_t<input_or_output_iterator<I>, int> = 0>
      constexpr void operator()(I& i, iter_difference_t<I> n) const
      {
        if constexpr (random_access_iterator<I>)
          i += n;
        else
        {
          while (n > 0)
          {
            --n;
            ++i;
          }

          if constexpr (bidirectional_iterator<I>)
          {
            while (n < 0)
            {
              ++n;
              --i;
            }
          }
        }
      }


      template<typename I, typename S, std::enable_if_t<input_or_output_iterator<I> and sentinel_for<S, I>, int> = 0>
      constexpr void operator()(I& i, S bound) const
      {
        if constexpr (std::is_assignable_v<I&, S>)
          i = std::move(bound);
        else if constexpr (sized_sentinel_for<S, I>)
          (*this)(i, bound - i);
        else
          while (i != bound)
            ++i;
      }


      template<typename I, typename S, std::enable_if_t<input_or_output_iterator<I> and sentinel_for<S, I>, int> = 0>
      constexpr iter_difference_t<I>
      operator()(I& i, iter_difference_t<I> n, S bound) const
      {
        if constexpr (sized_sentinel_for<S, I>)
        {
          // std::abs is not constexpr until C++23
          auto abs = [](const iter_difference_t<I> x) { return x < 0 ? -x : x; };

          if (const auto dist = abs(n) - abs(bound - i); dist < 0)
          {
            (*this)(i, bound);
            return -dist;
          }

          (*this)(i, n);
          return 0;
        }
        else
        {
          while (n > 0 && i != bound)
          {
            --n;
            ++i;
          }

          if constexpr (bidirectional_iterator<I>)
          {
            while (n < 0 && i != bound)
            {
              ++n;
              --i;
            }
          }

          return n;
        }
      }
    };
  }

  inline constexpr detail_advance::advance_fn advance;


  namespace detail_next
  {
    struct next_fn
    {
      template<typename I, std::enable_if_t<input_or_output_iterator<I>, int> = 0>
      constexpr I operator()(I i) const { ++i; return i; }

      template<typename I, std::enable_if_t<input_or_output_iterator<I>, int> = 0>
      constexpr I operator()(I i, iter_difference_t<I> n) const
      {
        stdcompat::ranges::advance(i, n);
        return i;
      }

      template<typename I, typename S, std::enable_if_t<input_or_output_iterator<I> and sentinel_for<S, I>, int> = 0>
      constexpr I operator()(I i, S bound) const
      {
        stdcompat::ranges::advance(i, bound);
        return i;
      }

      template<typename I, typename S, std::enable_if_t<input_or_output_iterator<I> and sentinel_for<S, I>, int> = 0>
      constexpr I operator()(I i, iter_difference_t<I> n, S bound) const
      {
        stdcompat::ranges::advance(i, n, bound);
        return i;
      }
    };
  }

  inline constexpr detail_next::next_fn next;


  // ---
  // iterator_t, const_iterator_t, sentinel_t, const_sentinel_t
  // ---

  template<typename R>
  using iterator_t = decltype(stdcompat::ranges::begin(std::declval<R&>()));

  template<typename R>
  using const_iterator_t = decltype(stdcompat::ranges::cbegin(std::declval<R&>()));


  template<typename R>
  using sentinel_t = decltype(stdcompat::ranges::end(std::declval<R&>()));

  template<typename R>
  using const_sentinel_t = decltype(stdcompat::ranges::cend(std::declval<R&>()));
#endif
}

#endif
