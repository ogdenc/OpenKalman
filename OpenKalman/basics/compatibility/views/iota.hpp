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
 * \brief Definition for \ref ranges::iota_view and \ref ranges::views::iota.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_IOTA_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_IOTA_HPP

#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/iterator.hpp"
#include "view-concepts.hpp"
#include "view_interface.hpp"

namespace OpenKalman::stdex::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::iota_view;
  namespace views
  {
    using std::ranges::views::iota;
  }
#else
  /**
   * \brief Equivalent to std::ranges::iota_view.
   */
  template<typename W, typename Bound = unreachable_sentinel_t>
  struct iota_view : view_interface<iota_view<W, Bound>>
  {
  private:

    static_assert(weakly_incrementable<W> and stdex::semiregular<Bound>);

    template<typename I>
    using iota_diff_t = std::conditional_t<
      not std::is_integral_v<I> or (sizeof(iter_difference_t<I>) > sizeof(I)), iter_difference_t<I>, std::ptrdiff_t>;

    template<typename I, typename = void>
    struct is_decrementable : std::false_type {};

    template<typename I>
    struct is_decrementable<I, std::enable_if_t<
      incrementable<I> and
      std::is_same<decltype(--std::declval<I&>()), I&>::value and
      std::is_same<decltype(std::declval<I&>()--), I>::value>> : std::true_type {};

    template<typename I>
    static constexpr bool decrementable = is_decrementable<I>::value;


    template<typename I, typename = void>
    struct is_advanceable : std::false_type {};

    template<typename I>
    struct is_advanceable<I, std::enable_if_t<
      decrementable<I> and
      stdex::convertible_to<decltype(std::declval<I>() == std::declval<I>()), bool> and
      stdex::convertible_to<decltype(std::declval<I>() != std::declval<I>()), bool> and
      stdex::convertible_to<decltype(std::declval<I>() < std::declval<I>()), bool> and
      stdex::convertible_to<decltype(std::declval<I>() > std::declval<I>()), bool> and
      stdex::convertible_to<decltype(std::declval<I>() <= std::declval<I>()), bool> and
      stdex::convertible_to<decltype(std::declval<I>() >= std::declval<I>()), bool> and
      std::is_same<decltype(std::declval<I&>() += std::declval<const iota_diff_t<I>>()), I&>::value and
      std::is_same<decltype(std::declval<I&>() -= std::declval<const iota_diff_t<I>>()), I&>::value and
      stdex::constructible_from<I, decltype(std::declval<const I&>() + std::declval<const iota_diff_t<I>>())> and
      stdex::constructible_from<I, decltype(std::declval<const iota_diff_t<I>>() + std::declval<const I&>())> and
      stdex::constructible_from<I, decltype(std::declval<const I&>() - std::declval<const iota_diff_t<I>>())> and
      std::is_convertible<decltype(std::declval<const I&>() - std::declval<const I&>()), iota_diff_t<I>>::value>> : std::true_type {};

    template<typename I>
    static constexpr bool advanceable = is_advanceable<I>::value;

  public:

    struct sentinel;


    struct iterator
    {
      using iterator_concept = std::conditional_t<advanceable<W>, std::random_access_iterator_tag,
        std::conditional_t<decrementable<W>, std::bidirectional_iterator_tag,
        std::conditional_t<incrementable<W>, std::forward_iterator_tag, std::input_iterator_tag>>>;

      using iterator_category = std::input_iterator_tag;

      using value_type = W;
      using difference_type = iota_diff_t<W>;
      using reference = W;
      using pointer = void;

      template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<W>, int> = 0>
      constexpr iterator() {};

      constexpr explicit iterator(W value) : value_ {value} {}

      constexpr W operator*() const noexcept(std::is_nothrow_copy_constructible_v<W>) { return value_; }

      constexpr iterator& operator++() { ++value_; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and not incrementable<W>, int> = 0>
      constexpr void operator++(int) { ++value_; }

      template<bool Enable = true, std::enable_if_t<Enable and incrementable<W>, int> = 0>
      constexpr iterator operator++(int) { auto tmp = *this; ++value_; return tmp; }

      constexpr iterator& operator--() { --value_; return *this; }

      constexpr iterator operator--(int) { auto tmp = *this; --value_; return tmp; }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      constexpr iterator& operator+=(const difference_type& n)
      {
        if constexpr (OpenKalman::internal::is_unsigned_integer_like<W>)
          { if (n >= 0) value_ += static_cast<W>(n); else value_ -= static_cast<W>(-n); }
        else
          { value_ += n; }
        return *this;
      }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      constexpr iterator& operator-=(const difference_type& n)
      {
        if constexpr (OpenKalman::internal::is_unsigned_integer_like<W>)
        { if (n >= 0) value_ -= static_cast<W>(n); else value_ += static_cast<W>(-n); }
        else
        { value_ -= n; }
        return *this;
      }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      constexpr value_type operator[](difference_type n) const noexcept { return W(value_ + n); }

      friend constexpr bool operator==(const iterator& x, const iterator& y) { return x.value_ == y.value_; }

      friend constexpr bool operator!=(const iterator& x, const iterator& y) { return not (x.value_ == y.value_); }

      friend constexpr bool operator<(const iterator& x, const iterator& y) { return x.value_ < y.value_; }

      friend constexpr bool operator>(const iterator& x, const iterator& y) { return y < x; }

      friend constexpr bool operator<=(const iterator& x, const iterator& y) { return not (y < x); }

      friend constexpr bool operator>=(const iterator& x, const iterator& y) { return not (x < y); }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      friend constexpr iterator operator+(iterator i, const difference_type& n) { i += n; return i; }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      friend constexpr iterator operator+(const difference_type& n, iterator i) { i += n; return i; }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      friend constexpr iterator operator-(iterator i, const difference_type& n) { i -= n; return i; }

      template<bool Enable = true, std::enable_if_t<Enable and advanceable<W>, int> = 0>
      friend constexpr difference_type operator-(const iterator& x, const iterator& y)
      {
        using D = difference_type;
        if constexpr (OpenKalman::internal::is_integer_like<W>)
        {
          if constexpr (OpenKalman::internal::is_signed_integer_like<W>) return D(D(x.value_) - D(y.value_));
          else return y.value_ > x.value_ ? D(-D(y.value_ - x.value_)) : D(x.value_ - y.value_);
        }
        else return x.value_ - y.value_;
      }

    private:

      friend struct iota_view::sentinel;
      W value_;

    };


    struct sentinel
    {
      sentinel() = default;

      constexpr explicit sentinel(Bound bound) : bound_ {bound} {};

      friend constexpr bool operator==( const iterator& x, const sentinel& y ) { return x.value_ == y.bound_; }

      friend constexpr bool operator==( const sentinel& y, const iterator& x ) { return y.bound_ == x.value_; }

      friend constexpr bool operator!=( const iterator& x, const sentinel& y ) { return not (x.value_ == y.bound_); }

      friend constexpr bool operator!=( const sentinel& y, const iterator& x ) { return not (y.bound_ == x.value_); }

      template<bool Enable = true, std::enable_if_t<Enable and stdex::convertible_to<decltype(std::declval<W>() - std::declval<Bound>()), iter_difference_t<W>>, int> = 0>
      friend constexpr iter_difference_t<W> operator-(const iterator& x, const sentinel& y) { return x.value_ - y.bound_; }

      template<bool Enable = true, std::enable_if_t<Enable and stdex::convertible_to<decltype(-(std::declval<W>() - std::declval<Bound>())), iter_difference_t<W>>, int> = 0>
      friend constexpr iter_difference_t<W> operator-(const sentinel& x, const iterator& y) { return -(y.value_ - x.bound_); }

    private:

      Bound bound_;
    };


    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<W>, int> = 0>
    constexpr
    iota_view() {};

    constexpr explicit
    iota_view(W value) : value_ {std::move(value)} {}

    constexpr explicit
    iota_view(stdex::type_identity_t<W> value, stdex::type_identity_t<Bound> bound) : value_ {value}, bound_ {bound} {}

    template<bool Enable = true, std::enable_if_t<Enable and std::is_same_v<Bound, W>, int> = 0>
    constexpr explicit
    iota_view(iterator first, iterator last) : value_ {first.value_}, bound_ {last.value_} {}

    template<bool Enable = true, std::enable_if_t<Enable and std::is_same_v<Bound, unreachable_sentinel_t>, int> = 0>
    constexpr explicit
    iota_view(iterator first, Bound last) : value_ {first.value_}, bound_ {last} {}

    template<bool Enable = true, std::enable_if_t<Enable and
      not std::is_same_v<Bound, W> and not std::is_same_v<Bound, unreachable_sentinel_t>, int> = 0>
    constexpr explicit
    iota_view(iterator first, sentinel last) : value_ {first.value_}, bound_ {last.bound_} {}


    constexpr iterator
    begin() const { return iterator {value_}; }


    constexpr auto
    end() const
    {
      if constexpr (std::is_same_v<Bound, unreachable_sentinel_t>) return unreachable_sentinel;
      else return sentinel {bound_};
    }

    template<bool Enable = true, std::enable_if_t<Enable and std::is_same_v<W, Bound>, int> = 0>
    constexpr iterator
    end() const { return sentinel {bound_}; }


    constexpr auto
    empty() const { return value_ == bound_; }

  private:

    template<typename A, typename B, typename = void>
    struct subtractable : std::false_type {};

    template<typename A, typename B>
    struct subtractable<A, B, std::enable_if_t<
      stdex::convertible_to<decltype(std::declval<A>() - std::declval<B>()), std::size_t>>> : std::false_type {};

  public:

    template<bool Enable = true, std::enable_if_t<Enable and
      not std::is_same_v<Bound, unreachable_sentinel_t> and
      ((std::is_same_v<W, Bound> and advanceable<W>) or
        (OpenKalman::internal::is_integer_like<W> and OpenKalman::internal::is_integer_like<Bound>) or
        sized_sentinel_for<Bound, W>), int> = 0>
    constexpr auto
    size() const
    {
      if constexpr (not OpenKalman::internal::is_integer_like<W> or not OpenKalman::internal::is_integer_like<Bound>)
        return static_cast<std::size_t>(bound_ - value_);
      else
        return value_ < 0 ?
        (bound_ < 0 ? static_cast<std::size_t>(-value_) - static_cast<std::size_t>(-bound_) :
          static_cast<std::size_t>(bound_) + static_cast<std::size_t>(-value_)) :
        static_cast<std::size_t>(bound_) - static_cast<std::size_t>(value_);
    }

  private:

    W value_;

    Bound bound_;

  }; // struct iota_view


  template<typename W, typename Bound, std::enable_if_t<
    not OpenKalman::internal::is_integer_like<W> or not OpenKalman::internal::is_integer_like<Bound> or
    OpenKalman::internal::is_signed_integer_like<W> == OpenKalman::internal::is_signed_integer_like<Bound>, int> = 0>
  iota_view(W, Bound) -> iota_view<W, Bound>;


  template<typename W, typename Bound>
  constexpr bool enable_borrowed_range<stdex::ranges::iota_view<W, Bound>> = true;


  namespace views
  {
    namespace detail
    {
      struct iota_adapter
      {
        template<typename W, typename Bound = unreachable_sentinel_t, std::enable_if_t<weakly_incrementable<W> and stdex::semiregular<Bound>, int> = 0>
        constexpr auto
        operator() [[nodiscard]] (W&& value, Bound&& bound = {}) const
        {
          return iota_view {std::forward<W>(value), std::forward<Bound>(bound)};
        }
      };
    }


    /**
     * \brief Equivalent to std::ranges::views::iota.
     * \sa iota_view
     */
    inline constexpr detail::iota_adapter iota;
  }

#endif
}

#endif
