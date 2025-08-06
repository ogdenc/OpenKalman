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
 * \brief Definition of \ref collections::replicate_view and \ref views::replicate.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP

#include <type_traits>
#include "basics/basics.hpp"
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_size.hpp"
#include "collections/functions/get.hpp"
#include "collections/functions/comparison_operators.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view that replicates a \ref collection some number of times.
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<replicate_view<std::tuple<int, double>, std::integral_constant<std::size_t, 3>>> == 6);
   * static_assert(std::is_same_v<std::tuple_element_t<5, replicate_view<std::tuple<double, int, float>, std::integral_constant<std::size_t, 2>>>, float>);
   * static_assert(get<3>(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}) == 5);
   * static_assert((replicate_view {std::vector{3, 4, 5}, 2u}[4]), 4);
   * static_assert((replicate_view {std::vector{3, 4, 5}, 2u}[std::integral_constant<std::size_t, 5>{}]), 5);
   * \endcode
   * \sa views::replicate
   */
#ifdef __cpp_lib_ranges
  template<collection V, values::index Factor> requires std::same_as<std::decay_t<Factor>, Factor>
#else
  template<typename V, typename Factor>
#endif
  struct replicate_view : stdcompat::ranges::view_interface<replicate_view<V, Factor>>
  {
  private:

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    /**
     * \brief Iterator for \ref replicate_view
     */
    template<bool Const>
    struct iterator
    {
    private:

      using Parent = maybe_const<Const, replicate_view>;

    public:

      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = stdcompat::ranges::range_value_t<V>;
      using reference = stdcompat::ranges::range_reference_t<V>;
      using difference_type = std::ptrdiff_t;
      using pointer = void;

      constexpr iterator() = default;

      constexpr iterator(Parent& parent, difference_type p) : parent_ {std::addressof(parent)}, current_{p} {}

      constexpr iterator(iterator<not Const> i) : parent_ {std::move(i.parent_)}, current_ {std::move(i.current_)} {}

      constexpr decltype(auto) operator*() { return collections::get(parent_->v_, current_ % get_size(parent_->v_)); }
      constexpr decltype(auto) operator*() const { return collections::get(parent_->v_, current_ % get_size(parent_->v_)); }
      constexpr decltype(auto) operator[](difference_type offset) { return collections::get(parent_->v_, (current_ + offset) % get_size(parent_->v_)); }
      constexpr decltype(auto) operator[](difference_type offset) const { return collections::get(parent_->v_, (current_ + offset) % get_size(parent_->v_)); }

      constexpr auto& operator++() noexcept { ++current_; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current_; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current_ += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current_ -= diff; return *this; }
      friend constexpr auto operator+(const iterator& it, const difference_type diff) noexcept
      { return iterator {*it.parent_, it.current_ + diff}; }
      friend constexpr auto operator+(const difference_type diff, const iterator& it) noexcept
      { return iterator {*it.parent_, diff + it.current_}; }
      friend constexpr auto operator-(const iterator& it, const difference_type diff)
      { return iterator {*it.parent_, it.current_ - diff}; }
      friend constexpr difference_type operator-(const iterator& it, const iterator& other) noexcept
      { return it.current_ - other.current_; }
      friend constexpr bool operator==(const iterator& it, const iterator& other) noexcept
      { return it.current_ == other.current_; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const iterator& other) const noexcept { return current_ <=> other.current_; }
#else
      constexpr bool operator!=(const iterator& other) const noexcept { return current_ != other.current_; }
      constexpr bool operator<(const iterator& other) const noexcept { return current_ < other.current_; }
      constexpr bool operator>(const iterator& other) const noexcept { return current_ > other.current_; }
      constexpr bool operator<=(const iterator& other) const noexcept { return current_ <= other.current_; }
      constexpr bool operator>=(const iterator& other) const noexcept { return current_ >= other.current_; }
#endif

    private:

      Parent * parent_;
      difference_type current_;

    };


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    replicate_view() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      stdcompat::default_initializable<V> and stdcompat::default_initializable<Factor>, int> = 0>
    constexpr
    replicate_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
    constexpr
    replicate_view(V& v, Factor f) : v_ {v}, f_ {std::move(f)} {}


    /// \overload
    constexpr
    replicate_view(V&& v, Factor f) : v_ {std::move(v)}, f_ {std::move(f)} {}


    /**
     * \brief The base view.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).v_; }
#else
    constexpr V& base() & { return this->v_; }
    constexpr const V& base() const & { return this->v_; }
    constexpr V&& base() && noexcept { return std::move(*this).v_; }
    constexpr const V&& base() const && noexcept { return std::move(*this).v_; }
#endif


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto
    begin() { return iterator<false> {*this, 0_uz}; }

#ifdef __cpp_concepts
    constexpr auto
    begin() const requires stdcompat::ranges::range<const V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::range<const V>, int> = 0>
    constexpr auto begin() const
#endif
    { return iterator<true> {*this, 0_uz}; }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto
    end() { return iterator<false> {*this, static_cast<std::ptrdiff_t>(values::to_number(size()))}; }


    /// \overload
#ifdef __cpp_concepts
    constexpr auto
    end() const requires stdcompat::ranges::range<const V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::ranges::range<const V>, int> = 0>
    constexpr auto
    end() const noexcept
#endif
    {
      return iterator<true> {*this, static_cast<std::ptrdiff_t>(values::to_number(size()))};
    }


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_concepts
    constexpr values::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return values::operation(std::multiplies{}, get_size(v_), f_);
    }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr (size_of_v<V> != dynamic_size and values::fixed<Factor>)
        static_assert(i < size_of_v<V> * values::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(std::forward<decltype(self)>(self).v_,
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(self.v_)));
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      if constexpr (size_of_v<V> != dynamic_size and values::fixed<Factor>)
        static_assert(i < size_of_v<V> * values::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(v_,
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(v_)));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr (size_of_v<V> != dynamic_size and values::fixed<Factor>)
        static_assert(i < size_of_v<V> * values::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(v_,
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(v_)));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr (size_of_v<V> != dynamic_size and values::fixed<Factor>)
        static_assert(i < size_of_v<V> * values::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(std::move(*this).v_,
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(std::move(*this).v_)));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr (size_of_v<V> != dynamic_size and values::fixed<Factor>)
        static_assert(i < size_of_v<V> * values::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(std::move(*this).v_,
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(std::move(*this).v_)));
    }
#endif

  private:

    V v_;
    Factor f_;
  };


  /**
   * \brief Deduction guide
   */
  template<typename V, typename F>
  replicate_view(const V&, const F&) -> replicate_view<V, F>;


} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdcompat::ranges
#endif
{
  template<typename V, typename F>
  constexpr bool enable_borrowed_range<OpenKalman::collections::replicate_view<V, F>> = enable_borrowed_range<V>;
}


#ifndef __cpp_lib_ranges
namespace OpenKalman::collections::detail
{
  template<typename V, typename F, typename = void>
  struct replicate_tuple_size {};

  template<typename V, typename F>
  struct replicate_tuple_size<V, F, std::enable_if_t<values::fixed<F> and (size_of<V>::value != dynamic_size)>>
    : std::integral_constant<std::size_t, size_of_v<V> * values::fixed_number_of_v<F>> {};


  template<std::size_t i, typename V, typename = void>
  struct replicate_tuple_element
  {
    using type = stdcompat::ranges::range_value_t<V>;
  };

  template<std::size_t i, typename V>
  struct replicate_tuple_element<i, V, std::enable_if_t<size_of<V>::value != dynamic_size>>
    : std::tuple_element<i % size_of_v<V>, std::decay_t<V>> {};

}
#endif


namespace std
{
#ifdef __cpp_lib_ranges
  template<typename V, OpenKalman::values::fixed F> requires (OpenKalman::collections::size_of_v<V> != OpenKalman::dynamic_size)
  struct tuple_size<OpenKalman::collections::replicate_view<V, F>>
    : integral_constant<std::size_t, std::tuple_size_v<std::decay_t<V>> * OpenKalman::values::fixed_number_of_v<F>> {};
#else
  template<typename V, typename F>
  struct tuple_size<OpenKalman::collections::replicate_view<V, F>>
    : OpenKalman::collections::detail::replicate_tuple_size<std::decay_t<V>, F> {};
#endif


#ifdef __cpp_lib_ranges
  template<size_t i, typename V, typename F> requires (OpenKalman::collections::size_of_v<V> != OpenKalman::dynamic_size)
  struct tuple_element<i, OpenKalman::collections::replicate_view<V, F>>
    : tuple_element<i % OpenKalman::collections::size_of_v<V>, decay_t<V>> {};

  template<size_t i, typename V, typename F> requires (OpenKalman::collections::size_of_v<V> == OpenKalman::dynamic_size)
  struct tuple_element<i, OpenKalman::collections::replicate_view<V, F>>
  {
    using type = OpenKalman::stdcompat::ranges::range_value_t<V>;
  };
#else
  template<size_t i, typename V, typename F>
  struct tuple_element<i, OpenKalman::collections::replicate_view<V, F>>
    : OpenKalman::collections::detail::replicate_tuple_element<i, V> {};
#endif

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    template<typename Factor>
    struct replicate_closure : stdcompat::ranges::range_adaptor_closure<replicate_closure<Factor>>
    {
      constexpr replicate_closure(Factor f) : factor_ {std::move(f)} {};

#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const
      {
        return replicate_view {all(std::forward<R>(r)), factor_};
      }

    private:
      Factor factor_;
    };


    struct replicate_adaptor
    {
#ifdef __cpp_concepts
      template<values::index Factor>
#else
      template<typename Factor, std::enable_if_t<values::index<Factor>, int> = 0>
#endif
      constexpr auto
      operator() (Factor factor) const
      {
        return replicate_closure<Factor> {std::move(factor)};
      }


#ifdef __cpp_concepts
      template<viewable_collection R, values::index Factor>
#else
      template<typename R, typename Factor, std::enable_if_t<viewable_collection<R> and values::index<Factor>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r, Factor factor) const
      {
        return replicate_view {all(std::forward<R>(r)), std::move(factor)};
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure associated with \ref replicate_view.
   * \details The expression <code>views::replicate{f}(arg)</code> is expression-equivalent
   * to <code>replicate_view(arg, f)</code> for any suitable \ref viewable_collection arg.
   * \sa replicate_view
   */
  inline constexpr detail::replicate_adaptor replicate;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP
