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
 * \brief Definition for \ref collection::iota_view and \ref collection::views::iota.
 */

#ifndef OPENKALMAN_IOTA_COLLECTION_HPP
#define OPENKALMAN_IOTA_COLLECTION_HPP

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

namespace OpenKalman::collection
{
#ifndef __cpp_lib_ranges_iota
  namespace detail
  {
    template<typename W>
    struct IotaRange
#ifdef __cpp_lib_ranges
      : std::ranges::view_interface<IotaRange>
#endif
    {
      struct Iterator
      {
        using iterator_category = std::random_access_iterator_tag;
        using value_type = W;
        using difference_type = std::decay_t<decltype(std::declval<value_type>() - std::declval<value_type>())>;
        explicit constexpr Iterator(value_type p) : pos{p} {}
        constexpr Iterator(const Iterator& other) = default;
        constexpr Iterator(Iterator&& other) noexcept = default;
        constexpr Iterator& operator=(const Iterator& other) = default;
        constexpr Iterator& operator=(Iterator&& other) noexcept = default;
        constexpr value_type operator*() const { return pos; }
        constexpr auto& operator++() noexcept { ++pos; return *this; }
        constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
        constexpr auto& operator--() noexcept { --pos; return *this; }
        constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
        constexpr auto& operator+=(const difference_type diff) noexcept { pos += diff; return *this; }
        constexpr auto& operator-=(const difference_type diff) noexcept { pos -= diff; return *this; }
        constexpr auto operator+(const difference_type diff) const noexcept { return Iterator {pos + diff}; }
        friend constexpr auto operator+(const difference_type diff, const Iterator& it) noexcept { return Iterator {diff + it.pos}; }
        constexpr auto operator-(const difference_type diff) const noexcept { return Iterator {pos - diff}; }
        constexpr value_type operator[](difference_type offset) const { return pos + offset; }
        constexpr bool operator==(const Iterator& other) const noexcept { return pos == other.pos; }
#ifdef __cpp_impl_three_way_comparison
        constexpr auto operator<=>(const Iterator& other) const noexcept { return pos <=> other.pos; }
#else
        constexpr bool operator!=(const Iterator& other) const noexcept { return pos != other.pos; }
        constexpr bool operator<(const Iterator& other) const noexcept { return pos < other.pos; }
        constexpr bool operator>(const Iterator& other) const noexcept { return pos > other.pos; }
        constexpr bool operator<=(const Iterator& other) const noexcept { return pos <= other.pos; }
        constexpr bool operator>=(const Iterator& other) const noexcept { return pos >= other.pos; }
#endif

      private:

        value_type pos;

      }; // struct Iterator

      explicit constexpr IotaRange(W w, W size) : my_w {w}, my_size {size} {}
      [[nodiscard]] constexpr auto begin() const { return Iterator {my_w}; }
      [[nodiscard]] constexpr auto end() const { return Iterator {my_w + my_size}; }
      [[nodiscard]] constexpr auto size() const { return my_size; }

    private:

      W my_w;
      W my_size;

    }; // struct IotaRange

  } // namespace detail
#endif


  /**
   * \brief An iota \ref collection that is a std::range and may also be \ref tuple_like.
   * \details In all cases, the result will be a std::range.
   * If the Size parameter is \ref value::fixed, then the result will also be
   * a \ref tuple_like sequence effectively in the form of
   * <code>std::integral_sequence<std::size_t, 0>{},...,std::integral_sequence<std::size_t, N>{}</code>
   * \tparam W The type of the iota.
   * \tparam Size The size of the resulting collection
   */
#ifdef __cpp_concepts
  template<value::index W, value::index Size = W> requires std::is_object_v<W> and
    std::convertible_to<value::number_type_of_t<Size>, value::number_type_of_t<W>>
#else
  template<typename W, typename Size = W>
#endif
  struct iota_view : collection_view_interface<iota_view<W, Size>>
#ifdef __cpp_lib_ranges_iota
    : std::ranges::iota_view<value::number_type_of_t<W>, value::number_type_of_t<W>>
#else
    : detail::IotaRange<number_type_of_t<W>>
#endif
  {
  private:

#ifdef __cpp_lib_ranges_iota
    using base = std::ranges::iota_view<value::number_type_of_t<W>, value::number_type_of_t<W>>;
#else
    using base = detail::IotaRange<number_type_of_t<W>>;
#endif

  public:

#ifdef __cpp_concepts
    constexpr iota_collection_view() requires value::fixed<W> and value::fixed<Size> = default;
#else
    template<typename aW = W, std::enable_if_t<fixed<aW> and fixed<Size>, int> = 0>
    constexpr iota_collection_view() : base {W{}, Size{}} {};
#endif


    constexpr iota_collection_view(const W& w, const Size& size)
      : base {value::to_number(w), value::to_number(w) + static_cast<value::number_type_of_t<W>>(to_number(size))} {}


    [[nodiscard]] constexpr auto size() const
    {
      if constexpr (value::fixed<Size>) return value::to_number(std::decay_t<Size>{});
      else return base::size();
    }

  }; // struct iota_collection_view


  template<typename W, typename Size>
  iota_collection_view(const W&, const Size&) -> iota_collection_view<W, Size>;


#ifdef __cpp_concepts
  template<size_t i, typename W, typename Size> requires fixed<Size>
#else
  template<size_t i, typename W, typename Size, std::enable_if_t<fixed<Size>, int> = 0>
#endif
  constexpr auto
  get(const iota_collection_view<W, Size>& v)
  {
    static_assert(i < fixed_number_of_v<Size>);
    if constexpr (fixed<W>) return operation {std::plus{}, W{}, std::integral_constant<std::size_t, i>{}};
    else return operation {std::plus{}, *v.begin(), std::integral_constant<std::size_t, i>{}};
  }


  /**
   * \brief Create an \ref iota_collection_view.
   * \tparam W The type of the iota.
   * \tparam Size The size of the resulting collection
   */
#ifdef __cpp_concepts
  template<index W, index Size> requires std::convertible_to<number_type_of_t<Size>, number_type_of_t<W>>
  constexpr collection auto
#else
  template<typename W, typename Size, std::enable_if_t<index<W> and index<Size> and
    std::is_convertible_v<number_type_of_t<Size>, number_type_of_t<W>>, int> = 0>
  constexpr auto
#endif
  iota_collection(W&& w, Size&& size)
  {
    return iota_collection_view {std::forward<W>(w), std::forward<Size>(size)};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Size, typename = void>
    struct iota_collection_view_tuple_size_impl {};

    template<typename Size>
    struct iota_collection_view_tuple_size_impl<Size, std::enable_if_t<fixed<Size>>>
      : std::integral_constant<size_t, fixed_number_of_v<Size>> {};


    template<std::size_t i, typename W, typename Size, typename = void>
    struct iota_collection_view_tuple_element_impl {};

    template<std::size_t i, typename W, typename Size>
    struct iota_collection_view_tuple_element_impl<i, W, Size, std::enable_if_t<fixed<Size>>>
    {
      static_assert(i < fixed_number_of_v<Size>);
      using type = operation<std::plus<>, std::decay_t<W>, std::integral_constant<std::size_t, i>>;
    };
  } // namespace detail
#endif

} // OpenKalman::value


namespace std
{
#ifdef __cpp_concepts
  template<typename W, OpenKalman::value::fixed Size>
  struct tuple_size<OpenKalman::value::iota_collection_view<W, Size>>
    : std::integral_constant<size_t, OpenKalman::value::fixed_number_of_v<Size>> {};
#else
  template<typename W, typename Size>
  struct tuple_size<OpenKalman::value::iota_collection_view<W, Size>>
    : OpenKalman::value::detail::iota_collection_view_tuple_size_impl<Size> {};
#endif


#ifdef __cpp_concepts
  template<std::size_t i, typename W, OpenKalman::value::fixed Size>
  struct tuple_element<i, OpenKalman::value::iota_collection_view<W, Size>>
  {
    static_assert(i < OpenKalman::value::fixed_number_of_v<Size>);
    using type = OpenKalman::value::operation<std::plus<>, std::decay_t<W>, std::integral_constant<std::size_t, i>>;
  };
#else
  template<std::size_t i, typename W, typename Size>
  struct tuple_element<i, OpenKalman::value::iota_collection_view<W, Size>>
    : OpenKalman::value::detail::iota_collection_view_tuple_element_impl<i, W, Size> {};
#endif

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct iota_impl
    {
#ifdef __cpp_concepts
      template<value::index W, value::index Size>
#else
      template<typename W, typename Size, std::enable_if_t<value::index<W> and value::index<Size>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (W&& w, Size&& size) const { return iota_view<W, Size> {std::forward<W>(w), std::forward<Size>(size)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref identity_view.
   * \details The expression <code>views::identity(arg)</code> is expression-equivalent
   * to <code>identity_view(arg)</code> for any suitable \ref collection arg.
   * \sa identity_view
   */
  inline constexpr detail::iota_impl iota;

}


#endif //OPENKALMAN_IOTA_COLLECTION_HPP
