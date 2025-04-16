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
 * \brief Definition of \ref collections::all_view and \ref collections::views::all.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_ALL_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_ALL_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "basics/classes/movable_wrapper.hpp"
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "collection_view_interface.hpp"
#include "internal/maybe_tuple_size.hpp"
#include "internal/maybe_tuple_element.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view to all members of a \ref collection. It may either own the collection or just reference it.
   * \details The view is only copyable if it is non-owning.
   * The following should compile:
   * \code
   * static_assert(equal_to{}(all_view{std::tuple{4, 5.}}, std::tuple{4, 5.}));
   * static_assert(equal_to{}(std::tuple{4, 5.}, all_view{std::tuple{4, 5.}}));
   * static_assert(equal_to{}(all_view{std::vector{4, 5, 6}}, std::vector{4, 5, 6}));
   * static_assert(equal_to{}(std::array{4, 5, 6}, all_view{std::array{4, 5, 6}}));
   * \endcode
   * \sa views::all
   */
#ifdef __cpp_lib_ranges
  template<collection T>
#else
  template<typename T>
#endif
  struct all_view : collection_view_interface<all_view<T>>
  {
  private:

    using MyT = OpenKalman::internal::movable_wrapper<T>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    all_view() requires std::default_initializable<MyT> = default;
#else
    template<typename aT = MyT, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr
    all_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<MyT, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&>, int> = 0>
#endif
    explicit constexpr
    all_view(Arg&& arg) : my_t {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from a \ref collection.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<MyT, Arg&&> and std::is_move_assignable_v<MyT>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&> and std::is_move_assignable<MyT>::value, int> = 0>
#endif
    constexpr all_view&
    operator=(Arg&& arg) { my_t = MyT{std::forward<Arg>(arg)}; return *this; }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i> requires tuple_like<T>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::forward<decltype(self)>(self).my_t.get());
    }
#else
    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() &
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(my_t.get());
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(my_t.get());
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).my_t.get());
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).my_t.get());
    }
#endif


    /**
     * \brief The size of the base object.
     */
#ifdef __cpp_concepts
    constexpr value::index auto
    size() const noexcept
#else
    constexpr auto size() const noexcept
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
      return std::ranges::begin(std::forward<decltype(self)>(self).my_t.get());
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() & { return ranges::begin(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() const & { return ranges::begin(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() && noexcept { return ranges::begin(std::move(*this).my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() const && noexcept { return ranges::begin(std::move(*this).my_t.get()); }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    end(this auto&& self) noexcept requires sized_random_access_range<T>
    {
      return std::ranges::end(std::forward<decltype(self)>(self).my_t.get());
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() & { return ranges::end(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() const & { return ranges::end(my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() && noexcept { return ranges::end(std::move(*this).my_t.get()); }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() const && noexcept { return ranges::end(std::move(*this).my_t.get()); }
#endif

  private:

    MyT my_t;

  };


  /**
   * \brief Deduction guide.
   */
  template<typename Arg>
  all_view(Arg&&) -> all_view<Arg>;


} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename T>
  constexpr bool enable_borrowed_range<OpenKalman::collections::all_view<T>> =
    std::is_lvalue_reference_v<T> or enable_borrowed_range<remove_cvref_t<T>>;
}


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::all_view<T>>
    : OpenKalman::collections::internal::maybe_tuple_size<std::decay_t<T>> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::all_view<T>>
    : OpenKalman::collections::internal::maybe_tuple_element<integral_constant<size_t, i>, T> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct all_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<all_closure>
#endif
    {
#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&& r) const { return all_view<R> {std::forward<R>(r)}; }
    };
  }


  /**
   * \brief a std::ranges::range_adaptor_closure associated with \ref all_view.
   * \details The expression <code>views::all(arg)</code> is expression-equivalent
   * to <code>all_view(arg)</code> for any suitable \ref collection arg.
   * \sa all_view
   */
  inline constexpr detail::all_closure all;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_ALL_HPP
