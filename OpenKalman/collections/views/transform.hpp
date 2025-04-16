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

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_TRANSFORM_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_TRANSFORM_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/invocable_on_collection.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "internal/maybe_tuple_size.hpp"
#include "internal/maybe_tuple_element.hpp"
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  namespace internal
  {
    /**
     * \internal
     * \brief Iterator for \ref transform_view
     * \tparam It The base iterator
     * \tparam F The transformation function
     */
    template<typename It, typename F>
    struct transform_view_iterator
    {
      using iterator_category = std::random_access_iterator_tag;
      using value_type = decltype(std::invoke(*std::declval<F*>(), *std::declval<It>()));
#ifdef __cpp_lib_ranges
      using difference_type = std::iter_difference_t<It>;
#else
      using difference_type = iter_difference_t<It>;
#endif
      constexpr transform_view_iterator(It it, F& f) : my_it {std::move(it)}, my_f {std::addressof(f)} {}
      constexpr transform_view_iterator() = default;
      constexpr transform_view_iterator(const transform_view_iterator& other) = default;
      constexpr transform_view_iterator(transform_view_iterator&& other) noexcept = default;
      constexpr transform_view_iterator& operator=(const transform_view_iterator& other) = default;
      constexpr transform_view_iterator& operator=(transform_view_iterator&& other) noexcept = default;
      constexpr value_type operator*() { return std::invoke(*my_f, *my_it); }
      constexpr value_type operator*() const { return std::invoke(*my_f, *my_it); }
      constexpr value_type operator[](difference_type offset) { return std::invoke(*my_f, my_it[offset]); }
      constexpr value_type operator[](difference_type offset) const { return std::invoke(*my_f, my_it[offset]); }
      constexpr auto& operator++() noexcept { ++my_it; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --my_it; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { my_it += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { my_it -= diff; return *this; }
      friend constexpr auto operator+(const transform_view_iterator& it, const difference_type diff) noexcept
      { return transform_view_iterator {it.my_it + diff, *it.my_f}; }
      friend constexpr auto operator+(const difference_type diff, const transform_view_iterator& it) noexcept
      { return transform_view_iterator {it.my_it + diff, *it.my_f}; }
      friend constexpr auto operator-(const transform_view_iterator& it, const difference_type diff)
      { return transform_view_iterator {it.my_it - diff, *it.my_f}; }
      friend constexpr difference_type operator-(const transform_view_iterator& it, const transform_view_iterator& other) noexcept
      { return it.my_it - other.my_it; }
      constexpr bool operator==(const transform_view_iterator& other) const noexcept { return my_it == other.my_it; }
  #ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const transform_view_iterator& other) const noexcept { return my_it <=> other.my_it; }
  #else
      constexpr bool operator!=(const transform_view_iterator& other) const noexcept { return my_it != other.my_it; }
      constexpr bool operator<(const transform_view_iterator& other) const noexcept { return my_it < other.my_it; }
      constexpr bool operator>(const transform_view_iterator& other) const noexcept { return my_it > other.my_it; }
      constexpr bool operator<=(const transform_view_iterator& other) const noexcept { return my_it <= other.my_it; }
      constexpr bool operator>=(const transform_view_iterator& other) const noexcept { return my_it >= other.my_it; }
  #endif

    private:

      It my_it;
      F* my_f;

    }; // struct Iterator


    template<typename It, typename F>
    transform_view_iterator(const It&, F&, std::size_t) -> transform_view_iterator<It, F>;

  }


  /**
   * \brief A \ref collection created by applying a transformation to another collection of the same size.
   * \tparam T An underlying collection to be transformed
   * \tparam F A callable object taking an element of T and resulting in another object
   */
#ifdef __cpp_lib_ranges
  template<collection T, invocable_on_collection<T> F>
#else
  template<typename T, typename F, typename = void>
#endif
  struct transform_view : collection_view_interface<transform_view<T, F>>
  {
  private:

    using MyT = OpenKalman::internal::movable_wrapper<T>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    transform_view() requires std::default_initializable<MyT> and std::default_initializable<F> = default;
#else
    template<typename aT = MyT, std::enable_if_t<std::is_default_constructible_v<aT> and std::is_default_constructible_v<F>, int> = 0>
    constexpr
    transform_view() {}
#endif


    /**
     * \brief Construct from a \ref collection and a callable object.
     */
#ifdef __cpp_concepts
    template<typename Arg, typename Func> requires std::constructible_from<MyT, Arg&&> and std::constructible_from<F, Func&&>
#else
    template<typename Arg, typename Func, std::enable_if_t<std::is_constructible_v<MyT, Arg&&> and std::is_constructible_v<F, Func&&>, int> = 0>
#endif
    constexpr
    transform_view(Arg&& arg, Func&& f) : my_t {std::forward<Arg>(arg)}, my_f {std::forward<Func>(f)} {}


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i> requires tuple_like<T>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return std::invoke(
        std::forward<decltype(self)>(self).my_f,
        OpenKalman::internal::generalized_std_get<i>(std::forward<decltype(self)>(self).my_t.get()));
    }
#else
    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() &
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return std::invoke(my_f, OpenKalman::internal::generalized_std_get<i>(my_t.get()));
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return std::invoke(my_f, OpenKalman::internal::generalized_std_get<i>(my_t.get()));
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return std::invoke(std::move(*this).my_f, OpenKalman::internal::generalized_std_get<i>(std::move(*this).my_t.get()));
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(size_of_v<T> == dynamic_size or i < size_of_v<T>, "Index out of range");
      return std::invoke(std::move(*this).my_f, OpenKalman::internal::generalized_std_get<i>(std::move(*this).my_t.get()));
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
      return internal::transform_view_iterator {
        std::ranges::begin(std::forward<decltype(self)>(self).my_t.get()),
        std::forward<decltype(self)>(self).my_f};
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() & { return internal::transform_view_iterator {ranges::begin(my_t.get()), my_f}; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() const & { return internal::transform_view_iterator {ranges::begin(my_t.get()), my_f}; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() && noexcept { return internal::transform_view_iterator {ranges::begin(std::move(*this).my_t.get()), std::move(*this).my_f}; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto begin() const && noexcept { return internal::transform_view_iterator {ranges::begin(std::move(*this).my_t.get()), std::move(*this).my_f}; }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    end(this auto&& self) noexcept requires sized_random_access_range<T>
    {
      return internal::transform_view_iterator {
        std::ranges::end(std::forward<decltype(self)>(self).my_t.get()),
        std::forward<decltype(self)>(self).my_f};
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() & { return internal::transform_view_iterator {ranges::end(my_t.get()), my_f}; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() const & { return internal::transform_view_iterator {ranges::end(my_t.get()), my_f}; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() && noexcept { return internal::transform_view_iterator {ranges::end(std::move(*this).my_t.get()), std::move(*this).my_f}; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT>, int> = 0>
    constexpr auto end() const && noexcept { return internal::transform_view_iterator {ranges::end(std::move(*this).my_t.get()), std::move(*this).my_f}; }
#endif

  private:

    MyT my_t;
    F my_f;

  };


  template<typename T, typename F>
  transform_view(T&&, F&&) -> transform_view<T, F>;


#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename C, typename F, typename = void>
    struct transform_view_tuple_element_impl {};

    template<std::size_t i, typename C, typename F>
    struct transform_view_tuple_element_impl<i, C, F, std::enable_if_t<tuple_like<C>>>
    {
      using type = std::invoke_result_t<F, std::tuple_element_t<i, C>>;
    };
  } // namespace detail
#endif

} // namespace OpenKalman::value


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename T, typename F>
  constexpr bool enable_borrowed_range<OpenKalman::collections::transform_view<T, F>> =
    (std::is_lvalue_reference_v<T> or enable_borrowed_range<remove_cvref_t<T>>) and std::is_lvalue_reference_v<F>;
}


namespace std
{
  template<typename T, typename F>
  struct tuple_size<OpenKalman::collections::transform_view<T, F>>
    : OpenKalman::collections::internal::maybe_tuple_size<std::decay_t<T>> {};


#ifdef __cpp_concepts
  template<std::size_t i, OpenKalman::collections::tuple_like T, typename F>
  struct tuple_element<i, OpenKalman::collections::transform_view<T, F>>
  {
    using type = std::invoke_result_t<F, std::tuple_element_t<i, T>>;
  };
#else
  template<std::size_t i, typename T, typename F>
  struct tuple_element<i, OpenKalman::collections::transform_view<T, F>>
    : OpenKalman::collections::detail::transform_view_tuple_element_impl<i, T, F> {};
#endif


  template<typename It, typename F>
  struct iterator_traits<OpenKalman::collections::internal::transform_view_iterator<It, F>>
  {
    using difference_type = typename OpenKalman::collections::internal::transform_view_iterator<It, F>::difference_type;
    using value_type = typename OpenKalman::collections::internal::transform_view_iterator<It, F>::value_type;
    using iterator_category = typename OpenKalman::collections::internal::transform_view_iterator<It, F>::iterator_category;
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    template<typename F>
    struct transform_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<transform_closure<F>>
#endif
    {
      constexpr transform_closure(F&& f) : my_f {std::forward<F>(f)} {};

#ifdef __cpp_concepts
      template<viewable_collection R> requires invocable_on_collection<F, R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R> and invocable_on_collection<F, R>, int> = 0>
#endif
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


#ifdef __cpp_concepts
      template<viewable_collection R, invocable_on_collection<R> F>
#else
      template<typename R, typename F, std::enable_if_t<viewable_collection<R> and invocable_on_collection<F, R>, int> = 0>
#endif
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

#endif //OPENKALMAN_COLLECTIONS_VIEWS_TRANSFORM_HPP
