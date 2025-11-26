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
 * \brief Definition for \ref collections::generate_view and \ref collections::views::generate.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_GENERATE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_GENERATE_HPP

#include <variant> // for std::monostate
#include "values/values.hpp"
#include "collections/functions/lexicographical_compare_three_way.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref collection_view created by lazily generating elements based on an index.
   * \details Typically, the generating function will be a closure.
   * \tparam F A callable object (possibly a closure) taking an index and producing an output value corresponding to the index.
   * This should preferably be able to take a \ref values::fixed index argument.
   * \tparam Size The size of the output collection. If it is <code>values::unbounded_size_t</code>, the view is unsized.
   */
#ifdef __cpp_concepts
  template<typename F, values::size Size = values::unbounded_size_t> requires
    std::same_as<Size, std::remove_reference_t<Size>> and std::is_object_v<F> and std::invocable<F&, std::size_t>
#else
  template<typename F, typename Size = values::unbounded_size_t>
#endif
  struct generate_view : stdex::ranges::view_interface<generate_view<F, Size>>
  {
  private:

    static_assert(not values::fixed_value_compares_with<Size, stdex::dynamic_extent>,
      "Size parameter for generate_view cannot be std::dynamic_extent");

    using F_box = internal::movable_wrapper<F>;

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    /**
     * \brief Iterator for \ref generate_view
     */
    template<bool Const>
    struct iterator
    {
      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = std::invoke_result_t<F_box&, std::size_t>;
      using difference_type = std::ptrdiff_t;
      using reference = value_type;
      using pointer = void;

      constexpr iterator() = default;

      constexpr iterator(maybe_const<Const, F_box>& f, std::size_t pos)
        : f_ {std::addressof(f)}, current_ {static_cast<difference_type>(pos)} {};

      constexpr iterator(iterator<not Const> i) : f_ {std::move(i.f_)}, current_ {std::move(i.current_)} {}

      constexpr value_type operator*() const
      {
        return stdex::invoke(*f_, static_cast<std::size_t>(current_));
      }

      constexpr value_type operator[](difference_type offset) const
      {
        return stdex::invoke(*f_, static_cast<std::size_t>(current_ + offset));
      }

      constexpr auto& operator++() noexcept { ++current_; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current_; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current_ += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current_ -= diff; return *this; }

      friend constexpr auto operator+(const iterator& it, const difference_type diff)
      { return iterator {*it.f_, static_cast<std::size_t>(it.current_ + diff)}; }
      friend constexpr auto operator+(const difference_type diff, const iterator& it)
      { return iterator {*it.f_, static_cast<std::size_t>(diff + it.current_)}; }
      friend constexpr auto operator-(const iterator& it, const difference_type diff)
      { if (it.current_ < diff) throw std::out_of_range{"Iterator out of range"}; return iterator {*it.f_, static_cast<std::size_t>(it.current_ - diff)}; }
      friend constexpr difference_type operator-(const iterator& it, const iterator& other)
      { return it.current_ - other.current_; }
      friend constexpr bool operator==(const iterator& it, const iterator& other)
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

      maybe_const<Const, F_box> * f_;
      difference_type current_;

    }; // struct Iterator


    /**
     * \brief Construct from a callable object and a size.
     */
    constexpr
    generate_view(F f, Size size)
      : f_box {std::move(f)}, size_ {std::move(size)} {}


    /**
     * \brief Construct from a callable object, if the view is unsized.
     */
#ifdef __cpp_concepts
    constexpr
    generate_view(F f) noexcept requires (not values::index<Size>)
#else
    template<bool Enable = true, std::enable_if_t<Enable and (not values::index<Size>), int> = 0>
    constexpr
    generate_view(F f) noexcept
#endif
      : f_box {std::move(f)} {}


    /**
     * \brief Default constructor.
     */
    constexpr
    generate_view() = default;


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto begin() { return iterator<false> {f_box, 0}; }

    /// \overload
    constexpr auto begin() const { return iterator<true> {f_box, 0}; }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto end()
    {
      if constexpr (values::index<Size>)
        return iterator<false> {f_box, static_cast<std::size_t>(size_)};
      else
        return stdex::unreachable_sentinel;
    }

    /// \overload
    constexpr auto end() const
    {
      if constexpr (values::index<Size>)
        return iterator<true> {f_box, static_cast<std::size_t>(size_)};
      else
        return stdex::unreachable_sentinel;
    }


    /**
     * \brief The size of the resulting object.
     */
#ifdef __cpp_concepts
    constexpr auto
    size() const noexcept requires values::index<Size>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (values::index<Size>), int> = 0>
    constexpr auto size() const noexcept
#endif
    {
      return size_;
    }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert (not values::fixed_value_compares_with<Size, i, &std::is_lteq>, "Index out of range");
      return std::forward<decltype(self)>(self).f_box(std::integral_constant<std::size_t, i>{});
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      static_assert (not values::fixed_value_compares_with<Size, i, &stdex::is_lteq>, "Index out of range");
      return f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      static_assert (not values::fixed_value_compares_with<Size, i, &stdex::is_lteq>, "Index out of range");
      return f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert (not values::fixed_value_compares_with<Size, i, &stdex::is_lteq>, "Index out of range");
      return std::move(*this).f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert (not values::fixed_value_compares_with<Size, i, &stdex::is_lteq>, "Index out of range");
      return std::move(*this).f_box(std::integral_constant<std::size_t, i>{});
    }
#endif

  private:

    F_box f_box;
    Size size_;

  };


  /**
   * \brief Deduction guide
   */
  template<typename F, typename S>
  generate_view(F, S) -> generate_view<F, S>;

  /// \overload
  template<typename F>
  generate_view(F) -> generate_view<F>;


}


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdex::ranges
#endif
{
  template<typename F, typename S>
  constexpr bool enable_borrowed_range<OpenKalman::collections::generate_view<F, S>> = std::is_lvalue_reference_v<F> or
    OpenKalman::stdex::semiregular<OpenKalman::collections::internal::movable_wrapper<F>>;
}


namespace std
{
  template<typename F, typename S>
  struct tuple_size<OpenKalman::collections::generate_view<F, S>> : OpenKalman::values::fixed_value_of<S> {};


  template<std::size_t i, typename F, typename S>
  struct tuple_element<i, OpenKalman::collections::generate_view<F, S>>
  {
    using type = std::invoke_result_t<F, std::integral_constant<std::size_t, i>>;
  };

}


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct generate_adaptor
    {
      /**
       * \brief Create a \ref generate_view.
       */
#ifdef __cpp_concepts
      template<typename F, values::index Size> requires
        std::invocable<F, std::size_t> and std::invocable<F, std::integral_constant<std::size_t, 0>>
#else
      template<typename F, typename Size, std::enable_if_t<values::index<Size> and
        std::is_invocable_v<F, std::size_t> and std::is_invocable_v<F, std::integral_constant<std::size_t, 0>>, int> = 0>
#endif
      constexpr auto
      operator() (F&& f, Size s) const
      {
        return generate_view {std::forward<F>(f), std::move(s)};
      }


      /**
       * \brief Create an unsized \ref generate_view.
       */
#ifdef __cpp_concepts
      template<typename F> requires
        std::invocable<F, std::size_t> and std::invocable<F, std::integral_constant<std::size_t, 0>>
#else
      template<typename F, std::enable_if_t<std::is_invocable_v<F, std::size_t> and
        std::is_invocable_v<F, std::integral_constant<std::size_t, 0>>, int> = 0>
#endif
      constexpr auto
      operator() (F&& f) const
      {
        return generate_view {std::forward<F>(f)};
      }

    };

  }


  /**
   * \brief a \ref collection_view generator associated with \ref generate_view.
   * \details The expression <code>views::generate(f, s)</code> is expression-equivalent to <code>generate_view(f, s)</code>.
   * \sa generate_view
   */
  inline constexpr detail::generate_adaptor generate;

}

#endif
