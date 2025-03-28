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
 * \brief Definition of \ref collections::identity_view and \ref collections::views::identity.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_IDENTITY_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_IDENTITY_HPP

#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "collection_view_interface.hpp"
#include "internal/tuple_size_base.hpp"
#include "internal/tuple_element_base.hpp"
#include "internal/MovableWrapper.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view that wraps a \ref collection and presents it as another \ref collection with unaltered components.
   * \details
   * The following should compile:
   * \code
   * static_assert(equal_to{}(identity_view{std::tuple{4, 5.}}, std::tuple{4, 5.}));
   * static_assert(equal_to{}(std::tuple{4, 5.}, identity_view{std::tuple{4, 5.}}));
   * static_assert(equal_to{}(identity_view{std::vector{4, 5, 6}}, std::vector{4, 5, 6}));
   * static_assert(equal_to{}(std::array{4, 5, 6}, identity_view{std::array{4, 5, 6}}));
   * \endcode
   * \sa views::identity
   */
#ifdef __cpp_lib_ranges
  template<collection T>
#else
  template<typename T>
#endif
  struct identity_view : collection_view_interface<identity_view<T>>
  {
  private:

    using MyT = internal::MovableWrapper<T>;

  public:

    /**
     * \brief Default constructor
     */
#ifdef __cpp_concepts
    constexpr
    identity_view() requires std::default_initializable<MyT> = default;
#else
    template<typename aT = MyT, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr
    identity_view() {}
#endif


    /**
     * \brief Construct from a \ref collection
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<MyT, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&>, int> = 0>
#endif
    explicit constexpr
    identity_view(Arg&& arg) : my_t {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from a \ref collection
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<MyT, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&>, int> = 0>
#endif
    constexpr identity_view&
    operator=(Arg&& arg) { my_t = MyT{std::forward<Arg>(arg)}; return *this; }


    /**
     * \brief Get element i
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) requires (size_of_v<T> == dynamic_size) or (i < size_of_v<T>)
    {
      return collections::get(std::forward<decltype(self)>(self).my_t.ref(), std::integral_constant<std::size_t, i>{});
    }
#else
    template<std::size_t i, std::enable_if_t<(size_of_v<T> == dynamic_size) or (i < size_of_v<T>), int> = 0>
    constexpr decltype(auto)
    get() &
    {
      return collections::get(my_t.ref(), std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i, std::enable_if_t<(size_of_v<T> == dynamic_size) or (i < size_of_v<T>), int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      return collections::get(my_t.ref(), std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i, std::enable_if_t<(size_of_v<T> == dynamic_size) or (i < size_of_v<T>), int> = 0>
    constexpr decltype(auto)
    get() &&
    {
      return collections::get(std::move(*this).my_t.ref(), std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i, std::enable_if_t<(size_of_v<T> == dynamic_size) or (i < size_of_v<T>), int> = 0>
    constexpr decltype(auto)
    get() const &&
    {
      return collections::get(std::move(*this).my_t.ref(), std::integral_constant<std::size_t, i>{});
    }
#endif


    #ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return get_collection_size(my_t.ref());
    }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_lib_ranges
    constexpr auto
    begin() const requires std::ranges::range<T> { return std::ranges::begin(my_t.ref()); }
#else
    template<typename aT = MyT, typename = std::void_t<decltype(ranges::begin(std::declval<aT>().ref()))>>
    constexpr auto
    begin() const { return ranges::begin(my_t.ref()); }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_lib_ranges
    constexpr auto
    end() const requires std::ranges::range<T> { return std::ranges::end(my_t.ref()); }
#else
    template<typename aT = MyT, typename = std::void_t<decltype(ranges::begin(std::declval<aT>().ref()))>>
    constexpr auto
    end() const { return ranges::end(my_t.ref()); }
#endif


#ifdef __cpp_impl_three_way_comparison
    constexpr auto operator<=>(const collection auto& other) const noexcept { return my_t.ref() <=> other; }
    constexpr bool operator==(const collection auto& other) const noexcept { return my_t.ref() == other; }
#else
    template<typename C, std::enable_if_t<collection<C>, int> = 0>
    constexpr bool operator==(const C& c) const noexcept { return my_t.ref() == c; }

    template<typename C, std::enable_if_t<collection<C> and not std::is_base_of_v<identity_view, C>, int> = 0>
    friend constexpr bool operator==(const C& c, const identity_view& v) noexcept { return c == v.my_t.ref(); }

    template<typename C, std::enable_if_t<collection<C>, int> = 0>
    constexpr bool operator!=(const C& c) const noexcept { return my_t.ref() != c; }

    template<typename C, std::enable_if_t<collection<C> and not std::is_base_of_v<identity_view, C>, int> = 0>
    friend constexpr bool operator!=(const C& c, const identity_view& v) noexcept { return c != v.my_t.ref(); }

    template<typename C, std::enable_if_t<collection<C>, int> = 0>
    constexpr bool operator<(const C& c) const noexcept { return my_t.ref() < c; }

    template<typename C, std::enable_if_t<collection<C> and not std::is_base_of_v<identity_view, C>, int> = 0>
    friend constexpr bool operator<(const C& c, const identity_view& v) noexcept { return c < v.my_t.ref(); }

    template<typename C, std::enable_if_t<collection<C>, int> = 0>
    constexpr bool operator>(const C& c) const noexcept { return my_t.ref() > c; }

    template<typename C, std::enable_if_t<collection<C> and not std::is_base_of_v<identity_view, C>, int> = 0>
    friend constexpr bool operator>(const C& c, const identity_view& v) noexcept { return c > v.my_t.ref(); }

    template<typename C, std::enable_if_t<collection<C>, int> = 0>
    constexpr bool operator<=(const C& c) const noexcept { return my_t.ref() <= c; }

    template<typename C, std::enable_if_t<collection<C> and not std::is_base_of_v<identity_view, C>, int> = 0>
    friend constexpr bool operator<=(const C& c, const identity_view& v) noexcept { return c <= v.my_t.ref(); }

    template<typename C, std::enable_if_t<collection<C>, int> = 0>
    constexpr bool operator>=(const C& c) const noexcept { return my_t.ref() >= c; }

    template<typename C, std::enable_if_t<collection<C> and not std::is_base_of_v<identity_view, C>, int> = 0>
    friend constexpr bool operator>=(const C& c, const identity_view& v) noexcept { return c >= v.my_t.ref(); }
#endif

  private:

    MyT my_t;

  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  identity_view(Arg&&) -> identity_view<Arg>;


} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename T>
  constexpr bool enable_borrowed_range<OpenKalman::collections::identity_view<T>> = borrowed_range<T>;
}


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::identity_view<T>>
    : OpenKalman::collections::internal::tuple_size_base<T> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::identity_view<T>>
    : OpenKalman::collections::internal::tuple_element_base<integral_constant<std::size_t, i>, T> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct identity_impl
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<identity_impl>
#endif
    {
#ifdef __cpp_concepts
      template<collection R>
#else
      template<typename R, std::enable_if_t<collection<R>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&& r) const { return identity_view<R> {std::forward<R>(r)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref identity_view.
   * \details The expression <code>views::identity(arg)</code> is expression-equivalent
   * to <code>identity_view(arg)</code> for any suitable \ref collection arg.
   * \sa identity_view
   */
  inline constexpr detail::identity_impl identity;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_IDENTITY_HPP
