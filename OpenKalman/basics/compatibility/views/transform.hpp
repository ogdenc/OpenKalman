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
 * \brief Definition for \ref collections::transform_view and \ref collections::views::transform.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_TRANSFORM_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_TRANSFORM_HPP

#ifndef __cpp_lib_ranges

#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/ranges/range-access.hpp"
#include "basics/compatibility/ranges/range-concepts.hpp"
#include "view-concepts.hpp"
#include "view_interface.hpp"
#include "all.hpp"

namespace OpenKalman::ranges
{
  /**
   * \brief Equivalent to std::ranges::transform_view.
   */
  template<typename V, typename F>
  struct transform_view : ranges::view_interface<transform_view<V, F>>
  {
  private:

    static_assert(ranges::input_range<V>);
    static_assert(std::is_move_constructible_v<F>);
    static_assert(ranges::view<V>);
    static_assert(std::is_object_v<F>);
    static_assert(std::is_invocable_v<F&, ranges::range_reference_t<V>>);

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    template<bool> struct sentinel;


    template<bool Const>
    struct iterator
    {
    private:

      using Parent = maybe_const<Const, transform_view>;
      using Base = maybe_const<Const, V>;

    public:

      using iterator_concept = std::conditional_t<
        ranges::random_access_range<Base>, std::random_access_iterator_tag,
        std::conditional_t<ranges::bidirectional_range<Base>, std::bidirectional_iterator_tag,
        std::conditional_t<ranges::forward_range<Base>, std::forward_iterator_tag, std::input_iterator_tag>>>;

      using reference = std::invoke_result_t<maybe_const<Const, F>&, ranges::range_reference_t<Base>>;
      using value_type = remove_cvref_t<reference>;
      using difference_type = ranges::range_difference_t<Base>;
      using pointer = void;

      using iterator_category = std::conditional_t<
        not std::is_reference_v<reference>,
        std::input_iterator_tag,
        typename std::iterator_traits<ranges::iterator_t<Base>>::iterator_category>;

      template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<ranges::iterator_t<Base>>, int> = 0>
      constexpr iterator() {};

      constexpr iterator(Parent& parent, ranges::iterator_t<Base> current) : current_ {std::move(current)}, parent_ {std::addressof(parent)} {}

      template<bool Enable = Const, std::enable_if_t<Enable and std::is_convertible_v<ranges::iterator_t<V>, ranges::iterator_t<Base>>, int> = 0>
      constexpr iterator(iterator<not Const> i) : current_ {std::move(i.current_)}, parent_ {std::move(i.parent_)} {}

      constexpr const ranges::iterator_t<Base>& base() const & noexcept { return current_; }

      constexpr ranges::iterator_t<Base> base() && { return std::move(current_); }

      constexpr decltype(auto) operator*() const
      {
#if __cplusplus >= 202002L
        namespace ns = std;
#else
        namespace ns = OpenKalman;
#endif
        return ns::invoke(std::get<0>(parent_->fun_), *current_);
      }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      constexpr decltype(auto) operator[](difference_type n) const
      {
#if __cplusplus >= 202002L
        namespace ns = std;
#else
        namespace ns = OpenKalman;
#endif
        return ns::invoke(std::get<0>(parent_->fun_), current_[n]);
      }

      constexpr iterator& operator++() { ++current_; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and not ranges::forward_range<Base>, int> = 0>
      constexpr void operator++(int) { ++current_; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::forward_range<Base>, int> = 0>
      constexpr iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::bidirectional_range<Base>, int> = 0>
      constexpr iterator& operator--() { --current_; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::bidirectional_range<Base>, int> = 0>
      constexpr iterator operator--(int) { auto tmp = *this; --*this; return tmp; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      constexpr iterator& operator+=(const difference_type& n) { current_ += n; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      constexpr iterator& operator-=(const difference_type& n) { current_ -= n; return *this; }

      friend constexpr bool operator==(const iterator& x, const iterator& y) { return x.current_ == y.current_; }

      friend constexpr bool operator!=(const iterator& x, const iterator& y) { return not (x.current_ == y.current_); }

      friend constexpr bool operator<(const iterator& x, const iterator& y) { return x.current_ < y.current_; }

      friend constexpr bool operator>(const iterator& x, const iterator& y) { return y < x; }

      friend constexpr bool operator<=(const iterator& x, const iterator& y) { return not (y < x); }

      friend constexpr bool operator>=(const iterator& x, const iterator& y) { return not (x < y); }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      friend constexpr iterator operator+(const iterator& i, const difference_type& n) { return iterator {*i.parent_, i.current_ + n}; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      friend constexpr iterator operator+(const difference_type& n, const iterator& i) { return iterator {*i.parent_, i.current_ + n}; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      friend constexpr iterator operator-(const iterator& i, const difference_type& n) { return iterator {*i.parent_, i.current_ - n}; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      friend constexpr difference_type operator-(const iterator& x, const iterator& y) { return x.current_ - y.current_; }

      template<bool Enable = true, std::enable_if_t<Enable and ranges::random_access_range<Base>, int> = 0>
      friend constexpr decltype(auto) iter_move(const iterator& i) noexcept(noexcept(std::invoke(std::get<0>(i.parent_->fun_), *i.current_)))
      {
        if constexpr (std::is_lvalue_reference_v<decltype(*i)>) return std::move(*i);
        else return *i;
      }

    private:

      template<bool> friend struct transform_view::sentinel;
      ranges::iterator_t<Base> current_;
      Parent* parent_;
    };


    template<bool Const>
    struct sentinel
    {
    private:

      using Parent = maybe_const<Const, transform_view>;
      using Base = maybe_const<Const, V>;
      using difference_type = ranges::range_difference_t<Base>;

    public:

      sentinel() = default;

      constexpr explicit sentinel(ranges::sentinel_t<Base> end) : end_ {end} {};

      constexpr const ranges::sentinel_t<Base>& base() const { return end_; }

      friend constexpr bool operator==( const iterator<Const>& x, const sentinel<Const>& y ) { return x.current_ == y.end_; }

      friend constexpr bool operator==( const sentinel<Const>& y, const iterator<Const>& x ) { return y.end_ == x.current_; }

      friend constexpr bool operator!=( const iterator<Const>& x, const sentinel<Const>& y ) { return not (x.current_ == y.end_); }

      friend constexpr bool operator!=( const sentinel<Const>& y, const iterator<Const>& x ) { return not (y.end_ == x.current_); }

      template<bool Enable = true, std::enable_if_t<Enable and
        std::is_convertible_v<decltype(std::declval<ranges::iterator_t<Base>>() - std::declval<ranges::sentinel_t<Base>>()), difference_type>, int> = 0>
      friend constexpr difference_type operator-(const iterator<Const>& x, const sentinel& y)
      { return x.current_ - y.end_; }

      template<bool Enable = true, std::enable_if_t<Enable and
        std::is_convertible_v<decltype(-(std::declval<ranges::sentinel_t<Base>>() - std::declval<ranges::iterator_t<Base>>())), difference_type>, int> = 0>
      friend constexpr difference_type operator-(const sentinel& y, const iterator<Const>& x)
      { return y.end_ - x.current_; }

    private:

      ranges::sentinel_t<Base> end_;
    };


    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<V> and std::is_default_constructible_v<F>, int> = 0>
    constexpr
    transform_view() {}


    constexpr explicit
    transform_view(V base, F fun) : base_ {std::move(base)}, fun_ {std::move(fun)} {}


    template<bool Enable = true, std::enable_if_t<Enable and std::is_copy_constructible_v<V>, int> = 0>
    constexpr V
    base() const& { return base_; }

    constexpr V
    base() && { return std::move(base_); }


    constexpr auto
    begin() { return iterator<false> {*this, ranges::begin(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and
      ranges::range<const V> and std::is_invocable_v<const F&, ranges::range_reference_t<const V>>, int> = 0>
    constexpr auto
    begin() const { return iterator<true> {*this, ranges::begin(base_)}; }


    template<bool Enable = true, std::enable_if_t<Enable and not ranges::common_range<V>, int> = 0>
    constexpr auto
    end() { return sentinel<false> {ranges::end(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and ranges::common_range<V>, int> = 0>
    constexpr auto
    end() { return iterator<false> {*this, ranges::end(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and ranges::range<const V> and not ranges::common_range<const V> and
      std::is_invocable_v<const F&, ranges::range_reference_t<const V>>, int> = 0>
    constexpr auto
    end() const { return sentinel<true> {ranges::end(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and ranges::common_range<const V> and
      std::is_invocable_v<const F&, ranges::range_reference_t<const V>>, int> = 0>
    constexpr auto
    end() const { return iterator<true> {*this, ranges::end(base_)}; }


    template<bool Enable = true, std::enable_if_t<Enable and ranges::sized_range<V>, int> = 0>
    constexpr auto
    size() { return ranges::size(base_); }

    template<bool Enable = true, std::enable_if_t<Enable and ranges::sized_range<const V>, int> = 0>
    constexpr auto
    size() const { return ranges::size(base_); }

  private:

    V base_;
    std::tuple<F> fun_;

  };


  template<typename R, typename F>
  transform_view(R&&, F) -> transform_view<views::all_t<R>, F>;

} // namespace OpenKalman::values


namespace OpenKalman::ranges::views
{
  namespace detail
  {
    template<typename F>
    struct transform_closure
      : ranges::range_adaptor_closure<transform_closure<F>>
    {
      constexpr transform_closure(F&& f) : my_f {std::forward<F>(f)} {};

      template<typename R, std::enable_if_t<ranges::viewable_range<R>, int> = 0>
      constexpr auto
      operator() (R&& r) const { return transform_view {std::forward<R>(r), my_f}; }

    private:
      F my_f;
    };


    struct transform_adaptor
    {
      template<typename F>
      constexpr auto
      operator() (F&& f) const
      {
        return transform_closure<F> {std::forward<F>(f)};
      }


      template<typename R, typename F, std::enable_if_t<ranges::viewable_range<R>, int> = 0>
      constexpr auto
      operator() (R&& r, F&& f) const
      {
        return transform_view {std::forward<R>(r), std::forward<F>(f)};
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure associated with \ref transform_view.
   * \details The expression <code>views::transform{f}(arg)</code> is expression-equivalent
   * to <code>transform_view(arg, f)</code> for any suitable \ref viewable_collection arg.
   * \sa transform_view
   */
  inline constexpr detail::transform_adaptor transform;

}

#endif

#endif //OPENKALMAN_COMPATIBILITY_VIEWS_TRANSFORM_HPP
