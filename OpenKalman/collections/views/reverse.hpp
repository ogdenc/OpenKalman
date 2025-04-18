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
 * \brief Definition of \ref collections::reverse_view and \ref collections::views::reverse.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_REVERSE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_REVERSE_HPP

#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "internal/maybe_tuple_size.hpp"
#include "internal/maybe_tuple_element.hpp"
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view that wraps a \ref collection and presents it as a \ref collection in reverse order.
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<reverse_view<std::tuple<int, double>>> == 2);
   * static_assert(std::tuple_size_v<reverse_view<std::tuple<>>> == 0);
   * static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<std::tuple<float, int, double>>>, double>);
   * static_assert(std::is_same_v<std::tuple_element_t<1, reverse_view<std::tuple<float, int, double>>>, int>);
   * static_assert(std::is_same_v<std::tuple_element_t<2, reverse_view<std::tuple<float, int, double>>>, float>);
   * static_assert(collections::get<0>(reverse_view {std::tuple{4, 5.}}) == 5.);
   * static_assert(collections::get<1>(reverse_view {std::tuple{4, 5.}}) == 4);
   * static_assert(collections::get<0>(reverse_view {std::tuple{4, std::monostate{}}}) == std::monostate{});
   * static_assert((reverse_view {std::vector{3, 4, 5}}[0u]) == 5);
   * static_assert((reverse_view {std::vector{3, 4, 5}}[1u]) == 4);
   * static_assert((reverse_view {std::vector{3, 4, 5}}[2u]) == 3);
   * \endcode
   * \sa views::reverse
   */
#ifdef __cpp_lib_ranges
  template<collection T>
#else
  template<typename T>
#endif
  struct reverse_view : collection_view_interface<reverse_view<T>>
  {
  private:

    using MyT = OpenKalman::internal::movable_wrapper<T>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr reverse_view() requires std::default_initializable<MyT> = default;
#else
    template<typename aT = MyT, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr reverse_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<MyT, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&>, int> = 0>
#endif
    explicit constexpr reverse_view(Arg&& arg) : my_t {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from a \ref collection.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<MyT, Arg&&> and std::is_move_assignable_v<MyT>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&> and std::is_move_assignable<MyT>::value, int> = 0>
#endif
    constexpr reverse_view&
    operator=(Arg&& arg) { my_t = MyT{std::forward<Arg>(arg)}; return *this; }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i> requires tuple_like<T>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      constexpr auto s = size_of_v<T>;
      static_assert(s == dynamic_size or i < s, "Index out of range");
      return OpenKalman::internal::generalized_std_get<s - i - 1_uz>(std::forward<decltype(self)>(self).my_t.get());
    }
#else
    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() &
    {
      constexpr auto s = size_of_v<T>;
      static_assert(s == dynamic_size or i < s, "Index out of range");
      return OpenKalman::internal::generalized_std_get<s - i - 1_uz>(my_t.get());
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      constexpr auto s = size_of_v<T>;
      static_assert(s == dynamic_size or i < s, "Index out of range");
      return OpenKalman::internal::generalized_std_get<s - i - 1_uz>(my_t.get());
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() && noexcept
    {
      constexpr auto s = size_of_v<T>;
      static_assert(s == dynamic_size or i < s, "Index out of range");
      return OpenKalman::internal::generalized_std_get<s - i - 1_uz>(std::move(*this).my_t.get());
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const && noexcept
    {
      constexpr auto s = size_of_v<T>;
      static_assert(s == dynamic_size or i < s, "Index out of range");
      return OpenKalman::internal::generalized_std_get<s - i - 1_uz>(std::move(*this).my_t.get());
    }
#endif


#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return get_collection_size(my_t.get());
    }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    begin(this auto&& self) noexcept requires sized_random_access_range<T>
    {
      return std::ranges::rbegin(std::forward<decltype(self)>(self).my_t.get());
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() & { return ranges::rbegin(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() const & { return ranges::rbegin(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() && noexcept { return ranges::rbegin(std::move(*this).my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() const && noexcept { return ranges::rbegin(std::move(*this).my_t.get()); }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    end(this auto&& self) noexcept requires sized_random_access_range<T>
    {
      return std::ranges::rend(std::forward<decltype(self)>(self).my_t.get());
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() & { return ranges::rend(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() const & { return ranges::rend(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() && noexcept { return ranges::rend(std::move(*this).my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() const && noexcept { return ranges::rend(std::move(*this).my_t.get()); }
#endif

  private:

    MyT my_t;
  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  reverse_view(Arg&&) -> reverse_view<Arg>;

} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename T>
  constexpr bool enable_borrowed_range<OpenKalman::collections::reverse_view<T>> =
    std::is_lvalue_reference_v<T> or enable_borrowed_range<remove_cvref_t<T>>;
}


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::reverse_view<T>>
    : OpenKalman::collections::internal::maybe_tuple_size<std::decay_t<T>> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::reverse_view<T>>
    : OpenKalman::collections::internal::maybe_tuple_element<
        OpenKalman::value::operation<std::minus<>, tuple_size<std::decay_t<T>>, integral_constant<std::size_t, i + 1_uz>>, T> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct reverse_impl
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<reverse_impl>
#endif
    {
#ifdef __cpp_concepts
      template<collection R>
#else
      template<typename R, std::enable_if_t<collection<R>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&& r) const { return reverse_view {std::forward<R>(r)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref reverse_view.
   * \details The expression <code>views::reverse(arg)</code> is expression-equivalent
   * to <code>reverse_view(arg)</code> for any suitable \ref collection arg.
   * \sa reverse_view
   */
  inline constexpr detail::reverse_impl reverse;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_REVERSE_HPP
