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

#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/invoke.hpp"
#include "basics/compatibility/internal/movable_box.hpp"
#include "view-concepts.hpp"
#include "view_interface.hpp"
#include "all.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::transform_view;
  namespace views
  {
    using std::ranges::views::transform;
  }
#else
  /**
   * \brief Equivalent to std::ranges::transform_view.
   */
  template<typename V, typename F>
  struct transform_view : stdcompat::ranges::view_interface<transform_view<V, F>>
  {
  private:

    static_assert(stdcompat::ranges::input_range<V>);
    static_assert(std::is_move_constructible_v<F>);
    static_assert(stdcompat::ranges::view<V>);
    static_assert(std::is_object_v<F>);
    static_assert(std::is_invocable_v<F&, stdcompat::ranges::range_reference_t<V>>);

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
      using MCF = maybe_const<Const, F>;

    public:

      using iterator_concept = std::conditional_t<
        stdcompat::ranges::random_access_range<Base>, std::random_access_iterator_tag,
        std::conditional_t<stdcompat::ranges::bidirectional_range<Base>, std::bidirectional_iterator_tag,
        std::conditional_t<stdcompat::ranges::forward_range<Base>, std::forward_iterator_tag, std::input_iterator_tag>>>;

      using iterator_category = std::conditional_t<
        not std::is_reference_v<std::invoke_result_t<MCF&, stdcompat::ranges::range_reference_t<Base>>>,
        std::input_iterator_tag,
        typename stdcompat::iterator_traits<stdcompat::ranges::iterator_t<Base>>::iterator_category>;

      using reference = std::invoke_result_t<MCF&, stdcompat::ranges::range_reference_t<Base>>;
      using value_type = stdcompat::remove_cvref_t<std::invoke_result_t<MCF&, stdcompat::ranges::range_reference_t<Base>>>;
      using difference_type = stdcompat::ranges::range_difference_t<Base>;
      using pointer = void;

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<stdcompat::ranges::iterator_t<Base>>, int> = 0>
      constexpr iterator() {};

      constexpr iterator(Parent& parent, stdcompat::ranges::iterator_t<Base> current) : current_ {std::move(current)}, parent_ {std::addressof(parent)} {}

      template<bool Enable = Const, std::enable_if_t<Enable and stdcompat::convertible_to<stdcompat::ranges::iterator_t<V>, stdcompat::ranges::iterator_t<Base>>, int> = 0>
      constexpr iterator(iterator<not Const> i) : current_ {std::move(i.current_)}, parent_ {std::move(i.parent_)} {}

      constexpr const stdcompat::ranges::iterator_t<Base>& base() const & noexcept { return current_; }

      constexpr stdcompat::ranges::iterator_t<Base> base() && { return std::move(current_); }

      constexpr decltype(auto) operator*() const
      {
        return stdcompat::invoke(*parent_->fun_, *current_);
      }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      constexpr decltype(auto) operator[](difference_type n) const
      {
        return stdcompat::invoke(*parent_->fun_, current_[n]);
      }

      constexpr iterator& operator++() { ++current_; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and not stdcompat::ranges::forward_range<Base>, int> = 0>
      constexpr void operator++(int) { ++current_; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::forward_range<Base>, int> = 0>
      constexpr iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::bidirectional_range<Base>, int> = 0>
      constexpr iterator& operator--() { --current_; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::bidirectional_range<Base>, int> = 0>
      constexpr iterator operator--(int) { auto tmp = *this; --*this; return tmp; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      constexpr iterator& operator+=(const difference_type& n) { current_ += n; return *this; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      constexpr iterator& operator-=(const difference_type& n) { current_ -= n; return *this; }

      friend constexpr bool operator==(const iterator& x, const iterator& y) { return x.current_ == y.current_; }

      friend constexpr bool operator!=(const iterator& x, const iterator& y) { return not (x.current_ == y.current_); }

      friend constexpr bool operator<(const iterator& x, const iterator& y) { return x.current_ < y.current_; }

      friend constexpr bool operator>(const iterator& x, const iterator& y) { return y < x; }

      friend constexpr bool operator<=(const iterator& x, const iterator& y) { return not (y < x); }

      friend constexpr bool operator>=(const iterator& x, const iterator& y) { return not (x < y); }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      friend constexpr iterator operator+(const iterator& i, const difference_type& n) { return iterator {*i.parent_, i.current_ + n}; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      friend constexpr iterator operator+(const difference_type& n, const iterator& i) { return iterator {*i.parent_, i.current_ + n}; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      friend constexpr iterator operator-(const iterator& i, const difference_type& n) { return iterator {*i.parent_, i.current_ - n}; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      friend constexpr difference_type operator-(const iterator& x, const iterator& y) { return x.current_ - y.current_; }

      template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::random_access_range<Base>, int> = 0>
      friend constexpr decltype(auto) iter_move(const iterator& i) noexcept(noexcept(stdcompat::invoke(*i.parent_->fun_, *i.current_)))
      {
        if constexpr (std::is_lvalue_reference_v<decltype(*i)>) return std::move(*i);
        else return *i;
      }

    private:

      template<bool> friend struct transform_view::sentinel;
      stdcompat::ranges::iterator_t<Base> current_;
      Parent* parent_;
    };


    template<bool Const>
    struct sentinel
    {
    private:

      using Parent = maybe_const<Const, transform_view>;
      using Base = maybe_const<Const, V>;
      using difference_type = stdcompat::ranges::range_difference_t<Base>;

    public:

      sentinel() = default;

      constexpr explicit sentinel(stdcompat::ranges::sentinel_t<Base> end) : end_ {end} {};

      constexpr const stdcompat::ranges::sentinel_t<Base>& base() const { return end_; }

      friend constexpr bool operator==( const iterator<Const>& x, const sentinel<Const>& y ) { return x.current_ == y.end_; }

      friend constexpr bool operator==( const sentinel<Const>& y, const iterator<Const>& x ) { return y.end_ == x.current_; }

      friend constexpr bool operator!=( const iterator<Const>& x, const sentinel<Const>& y ) { return not (x.current_ == y.end_); }

      friend constexpr bool operator!=( const sentinel<Const>& y, const iterator<Const>& x ) { return not (y.end_ == x.current_); }

      template<bool Enable = true, std::enable_if_t<Enable and
        stdcompat::convertible_to<decltype(std::declval<stdcompat::ranges::iterator_t<Base>>() - std::declval<stdcompat::ranges::sentinel_t<Base>>()), difference_type>, int> = 0>
      friend constexpr difference_type operator-(const iterator<Const>& x, const sentinel& y)
      { return x.current_ - y.end_; }

      template<bool Enable = true, std::enable_if_t<Enable and
        stdcompat::convertible_to<decltype(-(std::declval<stdcompat::ranges::sentinel_t<Base>>() - std::declval<stdcompat::ranges::iterator_t<Base>>())), difference_type>, int> = 0>
      friend constexpr difference_type operator-(const sentinel& y, const iterator<Const>& x)
      { return y.end_ - x.current_; }

    private:

      stdcompat::ranges::sentinel_t<Base> end_;
    };


    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<V> and stdcompat::default_initializable<F>, int> = 0>
    constexpr
    transform_view() {}


    constexpr explicit
    transform_view(V base, F fun) : base_ {std::move(base)}, fun_ {std::move(fun)} {}


    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::copy_constructible<V>, int> = 0>
    constexpr V
    base() const& { return base_; }

    constexpr V
    base() && { return std::move(base_); }


    constexpr auto
    begin() { return iterator<false> {*this, stdcompat::ranges::begin(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and
      stdcompat::ranges::range<const V> and std::is_invocable_v<const F&, stdcompat::ranges::range_reference_t<const V>>, int> = 0>
    constexpr auto
    begin() const { return iterator<true> {*this, stdcompat::ranges::begin(base_)}; }


    template<bool Enable = true, std::enable_if_t<Enable and not stdcompat::ranges::common_range<V>, int> = 0>
    constexpr auto
    end() { return sentinel<false> {stdcompat::ranges::end(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::common_range<V>, int> = 0>
    constexpr auto
    end() { return iterator<false> {*this, stdcompat::ranges::end(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::range<const V> and not stdcompat::ranges::common_range<const V> and
      std::is_invocable_v<const F&, stdcompat::ranges::range_reference_t<const V>>, int> = 0>
    constexpr auto
    end() const { return sentinel<true> {stdcompat::ranges::end(base_)}; }

    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::common_range<const V> and
      std::is_invocable_v<const F&, stdcompat::ranges::range_reference_t<const V>>, int> = 0>
    constexpr auto
    end() const { return iterator<true> {*this, stdcompat::ranges::end(base_)}; }


    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::sized_range<V>, int> = 0>
    constexpr auto
    size() { return stdcompat::ranges::size(base_); }

    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::sized_range<const V>, int> = 0>
    constexpr auto
    size() const { return stdcompat::ranges::size(base_); }

  private:

    V base_;
    OpenKalman::internal::movable_box<F> fun_;

  };


  template<typename R, typename F>
  transform_view(R&&, F) -> transform_view<views::all_t<R>, F>;


  namespace views
  {
    namespace detail
    {
      template<typename F>
      struct transform_closure
        : stdcompat::ranges::range_adaptor_closure<transform_closure<F>>
      {
        constexpr transform_closure(F&& f) : my_f {std::forward<F>(f)} {};

        template<typename R, std::enable_if_t<stdcompat::ranges::viewable_range<R>, int> = 0>
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


        template<typename R, typename F, std::enable_if_t<stdcompat::ranges::viewable_range<R>, int> = 0>
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
}

#endif
