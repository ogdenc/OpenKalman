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
 * \brief Definition for \ref collections::iota_view and \ref collections::views::iota.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_IOTA_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_IOTA_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/functions/to_number.hpp"
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  namespace internal
  {
    /**
     * \internal
     * \brief Iterator for \ref iota_view
     */
    template<typename W>
    struct iota_view_iterator
    {
      using iterator_category = std::random_access_iterator_tag;
      using value_type = W;
      using difference_type = std::ptrdiff_t;
      explicit constexpr iota_view_iterator(value_type p) : current{p} {}
      constexpr iota_view_iterator() = default;
      constexpr iota_view_iterator(const iota_view_iterator& other) = default;
      constexpr iota_view_iterator(iota_view_iterator&& other) noexcept = default;
      constexpr iota_view_iterator& operator=(const iota_view_iterator& other) = default;
      constexpr iota_view_iterator& operator=(iota_view_iterator&& other) noexcept = default;
      explicit constexpr operator value_type() const noexcept { return current; }
      constexpr value_type operator*() noexcept { return current; }
      constexpr value_type operator*() const noexcept { return current; }
      constexpr value_type operator[](difference_type offset) noexcept { return current + offset; }
      constexpr value_type operator[](difference_type offset) const noexcept { return current + offset; }
      constexpr auto& operator++() noexcept { ++current; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current -= diff; return *this; }
      friend constexpr auto operator+(const iota_view_iterator& it, const difference_type diff) noexcept
      { return iota_view_iterator {static_cast<value_type>(it.current + diff)}; }
      friend constexpr auto operator+(const difference_type diff, const iota_view_iterator& it) noexcept
      { return iota_view_iterator {static_cast<value_type>(diff + it.current)}; }
      friend constexpr auto operator-(const iota_view_iterator& it, const difference_type diff)
      { if (static_cast<difference_type>(it.current) < diff) throw std::out_of_range{"Iterator out of range"};
        return iota_view_iterator {static_cast<value_type>(it.current - diff)}; }
      friend constexpr difference_type operator-(const iota_view_iterator& it, const iota_view_iterator& other) noexcept
      { return it.current - other.current; }
      friend constexpr bool operator==(const iota_view_iterator& it, const iota_view_iterator& other) noexcept
      { return it.current == other.current; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const iota_view_iterator& other) const noexcept { return current <=> other.current; }
#else
      constexpr bool operator!=(const iota_view_iterator& other) const noexcept { return current != other.current; }
      constexpr bool operator<(const iota_view_iterator& other) const noexcept { return current < other.current; }
      constexpr bool operator>(const iota_view_iterator& other) const noexcept { return current > other.current; }
      constexpr bool operator<=(const iota_view_iterator& other) const noexcept { return current <= other.current; }
      constexpr bool operator>=(const iota_view_iterator& other) const noexcept { return current >= other.current; }
#endif

    private:

      value_type current;

    }; // struct Iterator


    template<typename W>
    iota_view_iterator(const W&) -> iota_view_iterator<value::number_type_of_t<W>>;

  } // namespace internal


  /**
   * \brief An iota \ref collection that is a std::range and may also be \ref tuple_like.
   * \details In all cases, the result will be a std::range.
   * If the Size parameter is \ref value::fixed, then the result will also be
   * a \ref tuple_like sequence effectively in the form of
   * <code>std::integral_sequence<std::size_t, 0>{},...,std::integral_sequence<std::size_t, N>{}</code>
   * \tparam Start The start value of the iota.
   * \tparam Size The size of the resulting collection
   */
#ifdef __cpp_concepts
  template<value::integral Start, value::index Size = value::number_type_of_t<Start>> requires
    std::convertible_to<value::number_type_of_t<Size>, value::number_type_of_t<Start>>
#else
  template<typename Start, typename Size = value::number_type_of_t<Start>>
#endif
  struct iota_view : collection_view_interface<iota_view<Start, Size>>
  {
    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr iota_view() requires value::fixed<Start> and value::fixed<Size> = default;
#else
    template<typename aW = Start, std::enable_if_t<value::fixed<aW> and value::fixed<Size>, int> = 0>
    constexpr iota_view() {};
#endif


    /**
     * \brief Construct from an initial value and size.
     */
    constexpr iota_view(const Start& start, const Size& size) : my_start {start}, my_size {size} {}


    /**
     * \brief Construct from a size, starting at 1.
     */
    constexpr iota_view(const Size& size)
      : my_start {std::integral_constant<value::number_type_of_t<Start>, 1>{}}, my_size {size} {}


    /**
     * \brief Get element i.
     */
    template<size_t i>
    constexpr auto
    get() const
    {
      if constexpr (value::fixed<Size>) static_assert(i < value::fixed_number_of_v<Size>);
      return value::operation {std::plus{}, my_start, std::integral_constant<std::size_t, i>{}};
    }


    /**
     * \returns The size of the object.
     */
    constexpr auto size() const
    {
      return my_size;
    }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto begin() { return internal::iota_view_iterator {my_start}; }

    /// \overload
    constexpr auto begin() const { return internal::iota_view_iterator {my_start}; }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto end() { return internal::iota_view_iterator {my_start + my_size}; }


    /// \overload
    constexpr auto end() const { return internal::iota_view_iterator {my_start + my_size}; }

  private:

    Start my_start;
    Size my_size;

  }; // struct iota_view


  template<typename Start, typename Size>
  iota_view(const Start&, const Size&) -> iota_view<Start, Size>;


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Size, typename = void>
    struct iota_view_tuple_size_impl {};

    template<typename Size>
    struct iota_view_tuple_size_impl<Size, std::enable_if_t<value::fixed<Size>>>
      : std::integral_constant<std::size_t, value::fixed_number_of_v<Size>> {};


    template<std::size_t i, typename Start, typename Size, typename = void>
    struct iota_view_tuple_element_impl {};

    template<std::size_t i, typename Start, typename Size>
    struct iota_view_tuple_element_impl<i, Start, Size, std::enable_if_t<value::fixed<Size>>>
    {
      static_assert(i < value::fixed_number_of_v<Size>);
      using type = value::operation<std::plus<>, std::decay_t<Start>, std::integral_constant<std::size_t, i>>;
    };
  } // namespace detail
#endif

} // OpenKalman::value


namespace std
{
#ifdef __cpp_concepts
  template<typename Start, OpenKalman::value::fixed Size>
  struct tuple_size<OpenKalman::collections::iota_view<Start, Size>>
    : std::integral_constant<size_t, OpenKalman::value::fixed_number_of_v<Size>> {};
#else
  template<typename Start, typename Size>
  struct tuple_size<OpenKalman::collections::iota_view<Start, Size>>
    : OpenKalman::collections::detail::iota_view_tuple_size_impl<Size> {};
#endif


#ifdef __cpp_concepts
  template<std::size_t i, OpenKalman::value::fixed Start, OpenKalman::value::fixed Size>
  struct tuple_element<i, OpenKalman::collections::iota_view<Start, Size>>
  {
    static_assert(i < OpenKalman::value::fixed_number_of_v<Size>);
    using type = OpenKalman::value::operation<std::plus<>, Start, std::integral_constant<std::size_t, i>>;
  };
#else
  template<std::size_t i, typename Start, typename Size>
  struct tuple_element<i, OpenKalman::collections::iota_view<Start, Size>>
    : OpenKalman::collections::detail::iota_view_tuple_element_impl<i, Start, Size> {};
#endif


  template<typename W>
  struct iterator_traits<OpenKalman::collections::internal::iota_view_iterator<W>>
  {
    using difference_type = typename OpenKalman::collections::internal::iota_view_iterator<W>::difference_type;
    using value_type = typename OpenKalman::collections::internal::iota_view_iterator<W>::value_type;
    using iterator_category = typename OpenKalman::collections::internal::iota_view_iterator<W>::iterator_category;
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct iota_adapter
    {
#ifdef __cpp_concepts
      template<value::index Start, value::index Size>
#else
      template<typename Start, typename Size, std::enable_if_t<value::index<Start> and value::index<Size>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (Start start, Size size) const
      {
        return collections::iota_view {std::move(start), std::move(size)};
      }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref iota_view.
   * \details The expression <code>views::iota(arg)</code> is expression-equivalent
   * to <code>iota_view(arg)</code> for any suitable \ref collection arg.
   * \sa iota_view
   */
  inline constexpr detail::iota_adapter iota;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_IOTA_HPP
