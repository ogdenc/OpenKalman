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
 * \brief Definition for \ref collections::update_view and \ref collections::views::update.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_UPDATE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_UPDATE_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "all.hpp"
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get_size.hpp"
#include "collections/functions/compare.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref collection_view that updates an underlying collection_view on an element-by-element basis.
   * \tparam V An underlying \ref collection_view to be updated
   * \tparam F A callable update object of the form
   * <code>[](V&amp;&amp; t, values::index auto i) -> std::convertible_to&lt;std::ranges::range_value_t&lt;V&gt;*gt;</code>,
   * which calculates an updated value of element i of V.
   */
#ifdef __cpp_lib_ranges
  template<collection_view V, typename F> requires
      std::is_invocable_r_v<std::ranges::range_value_t<const V>, F&, std::remove_cvref_t<V>&, std::integral_constant<std::size_t, 0>> 
  struct update_view : std::ranges::view_interface<update_view<V, F>>
#else
  template<typename V, typename F>
  struct update_view : ranges::view_interface<update_view<V, F>>
#endif
  {
  private:

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

    using F_box = internal::movable_wrapper<F>;

    using value_type = std::invoke_result_t<F, V&, std::size_t>;

  public:

    /**
     * \brief A proxy object for accessing the base view
     */
    template<bool Const, typename Index>
    class proxy
    {
      using Parent = maybe_const<Const, update_view>;

      template<typename Fun, typename...Args>
      static constexpr decltype(auto)
      invoke_fun(Fun&& f, Args&&... args) noexcept(std::is_nothrow_invocable_v<Fun, Args...>)
      {
  #if __cplusplus >= 202002L
        namespace ns = std;
  #else
        namespace ns = OpenKalman;
  #endif
        return ns::invoke(std::forward<Fun>(f), std::forward<Args>(args)...);
      }

    public:

      constexpr proxy(Parent * parent, Index current = {}) : parent_ {parent}, current_ {std::move(current)} {}

      constexpr operator std::ranges::range_value_t<V> () const
      {
        return invoke_fun(parent_->f_.get(), parent_->v_, current_);
      }

#ifdef __cpp_lib_ranges
      constexpr proxy& operator=(std::ranges::range_value_t<V> x) requires
        std::invocable<S&, std::remove_cvref_t<V>&, std::integral_constant<std::size_t, 0>, std::ranges::range_value_t<V>> and
        std::ranges::output_range<V, std::ranges::range_value_t<views::all_t<V>>>
#else
      template<bool Enable = true, std::enable_if_t<Enable and
        std::is_invocable_v<S&, remove_cvref_t<V>&, std::integral_constant<std::size_t, 0>, ranges::range_value_t<V>> and
        ranges::output_range<V, ranges::range_value_t<views::all_t<V>>>, int> = 0>
      constexpr proxy& operator=(std::ranges::range_value_t<V> x) requires
#endif
      {
        invoke_fun(parent_->s_.get(), parent_->v_, current_, std::move(x));
        return *this;
      }

      private:

        Parent * parent_;
        Index current_;
      };


    /**
     * \brief Iterator for \ref update_view
     * \tparam Const Whether the iterator is constant
     */
    template<bool Const>
    class iterator
    {
      using Parent = maybe_const<Const, update_view>;
      using Base = maybe_const<Const, V>;

    public:

      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using pointer = void;
      using value_type = proxy<Const, std::size_t>;
      using reference = value_type;

      constexpr iterator() = default;

      constexpr iterator(maybe_const<Const, Parent>* parent, std::size_t pos)
        : current_ {static_cast<difference_type>(pos)}, parent_ {parent} {}

      explicit constexpr iterator(iterator<not Const> i) : current_ {std::move(i.current_)}, parent_ {std::move(i.parent_)} {}

      constexpr auto operator*() const { return value_type {parent_, static_cast<std::size_t>(current_)}; }

      constexpr auto operator[](difference_type n) const { return value_type {parent_, static_cast<std::size_t>(current_ + n)}; }

      constexpr iterator& operator++() { ++current_; return *this; }

      constexpr iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }

      constexpr iterator& operator--() { --current_; return *this; }

      constexpr iterator operator--(int) { auto tmp = *this; --*this; return tmp; }

      constexpr iterator& operator+=(const difference_type& n) { current_ += n; return *this; }

      constexpr iterator& operator-=(const difference_type& n) { current_ -= n; return *this; }

      friend constexpr bool operator==(const iterator& x, const iterator& y) { return x.current_ == y.current_; }

      friend constexpr bool operator!=(const iterator& x, const iterator& y) { return not (x.current_ == y.current_); }

      friend constexpr bool operator<(const iterator& x, const iterator& y) { return x.current_ < y.current_; }

      friend constexpr bool operator>(const iterator& x, const iterator& y) { return y < x; }

      friend constexpr bool operator<=(const iterator& x, const iterator& y) { return not (y < x); }

      friend constexpr bool operator>=(const iterator& x, const iterator& y) { return not (x < y); }

      friend constexpr iterator operator+(const iterator& i, const difference_type& n) { return iterator {*i.parent_, i.current_ + n}; }

      friend constexpr iterator operator+(const difference_type& n, const iterator& i) { return iterator {*i.parent_, i.current_ + n}; }

      friend constexpr iterator operator-(const iterator& i, const difference_type& n) { return iterator {*i.parent_, i.current_ - n}; }

      friend constexpr difference_type operator-(const iterator& x, const iterator& y) { return x.current_ - y.current_; }

    private:

      difference_type current_;
      Parent * parent_;
    };


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    update_view() requires std::default_initializable<V> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<V> and
      std::is_default_constructible_v<F_box> and std::is_default_constructible_v<S_box>, int> = 0>
    constexpr
    update_view() {}
#endif


    /**
     * \brief Construct from a \ref collection, a getter function, and a setter function.
     */
    template<typename G_, typename S_>
    constexpr
    update_view(V& v, G_&& g, S_&& s) : v_ {v}, f_ {std::forward<G_>(g)}, s_ {std::forward<S_>(s)} {}

    /// \overload
    template<typename G_, typename S_>
    constexpr
    update_view(V&& v, G_&& g, S_&& s) : v_ {std::move(v)}, f_ {std::forward<G_>(g)}, s_ {std::forward<S_>(s)} {}


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto begin() { return iterator<false> {this, 0}; }

    /// \overload
    constexpr auto begin() const { return iterator<true> {this, 0}; }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto end()
    {
      using namespace std;
      if constexpr (sized<V>) return iterator<false> {this, get_size(v_)};
      else return unreachable_sentinel;
    }

    /// \overload
    constexpr auto end() const
    {
      using namespace std;
      if constexpr (sized<V>) return iterator<true> {this, get_size(v_)};
      else return unreachable_sentinel;
    }


    /**
     * \brief The size of the resulting object.
     */
#ifdef __cpp_concepts
    constexpr values::index auto
    size() const requires sized<V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and sized<V>, int> = 0>
    constexpr auto size() const
#endif
    {
      return get_size(v_);
    }


    /**
     * \brief Get element i.
     */
    template<std::size_t i>
    constexpr decltype(auto)
    get()
    {
      if constexpr (sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return proxy<false, std::integral_constant<std::size_t, i>> {this};
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto)
    get() const
    {
      if constexpr (sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return proxy<true, std::integral_constant<std::size_t, i>> {this};
    }

  private:

    V v_;
    F_box f_;

  };


  template<typename V, typename F>
  update_view(const V&, F&&) -> update_view<V, F>;


#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename C, typename Fun, typename = void>
    struct update_view_tuple_element_impl {};

    template<std::size_t i, typename C, typename Fun>
    struct update_view_tuple_element_impl<i, C, Fun, std::enable_if_t<tuple_like<C>>>
    {
      using type = std::invoke_result_t<Fun, std::tuple_element_t<i, C>>;
    };
  } // namespace detail
#endif



#ifndef __cpp_concepts
  namespace detail_update
  {
    template<typename V, typename = void>
    struct tuple_size {};

    template<typename V>
    struct tuple_size<V, std::enable_if_t<sized<V> and size_of<V>::value != dynamic_size>> : size_of<V> {};
  }
#endif

}


namespace std
{
#ifdef __cpp_concepts
  template<OpenKalman::collections::sized V, typename F> requires
    (OpenKalman::collections::size_of_v<V> != OpenKalman::dynamic_size)
  struct tuple_size<OpenKalman::collections::update_view<V, F>> : OpenKalman::collections::size_of<V> {};
#else
  template<typename V, typename F>
  struct tuple_size<OpenKalman::collections::update_view<V, F>> : OpenKalman::collections::detail_update::tuple_size<V> {};
#endif


  template<std::size_t i, typename V, typename F>
  struct tuple_element<i, OpenKalman::collections::update_view<V, F>>
  {
#ifdef __cpp_lib_ranges
    using type = std::invoke_result_t<F, V&, std::integral_constant<std::size_t, i>>;
#else
    using type = std::invoke_result_t<F, V&, std::integral_constant<std::size_t, i>>;
#endif
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    template<typename F>
    struct update_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<update_closure<F>>
#else
      : ranges::range_adaptor_closure<update_closure<F>>
#endif
    {
      constexpr update_closure(F&& f) : f_ {std::forward<F>(f)} {};


#ifdef __cpp_concepts
      template<viewable_collection R> requires
        std::is_invocable_r_v<std::ranges::range_value_t<R>, F, all_t<R&&>, std::integral_constant<std::size_t, 0>> 
#else
      template<typename R>
#endif
      constexpr auto
      operator() (R&& r) const { return update_view {all(std::forward<R>(r)), f_}; }

    private:

      F f_;
    };


    struct update_adaptor
    {
      template<typename F>
      constexpr auto
      operator() (F&& f) const
      {
        return update_closure<std::decay_t<F>> {std::forward<F>(f)};
      }


#ifdef __cpp_concepts
      template<viewable_collection R, typename F> requires
        std::is_invocable_r_v<std::ranges::range_value_t<R>, F, all_t<R&&>, std::integral_constant<std::size_t, 0>>
#else
      template<typename R, typename F>
#endif
      constexpr auto
      operator() (R&& r, F&& g) const
      {
        return update_view {std::forward<R>(r) | all, std::forward<F>(g)};
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure associated with \ref update_view.
   * \details The expression <code>views::update(arg, f)</code> is expression-equivalent
   * to <code>update_view(views::all(arg), f)</code> for any suitable \ref viewable_collection arg.
   * \sa update_view
   */
  inline constexpr detail::update_adaptor update;

}

#endif //OPENKALMAN_COLLECTIONS_VIEWS_UPDATE_HPP
