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
 * \brief Definition of \ref collections::slice_view and \ref collections::views::slice.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP

#include <tuple>
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "values/functions/cast_to.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "collection_view_interface.hpp"
#include "internal/maybe_tuple_size.hpp"
#include "internal/maybe_tuple_element.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view representing a slice of a \ref collection.
   * \details
   * The following should compile:
   * \code
   * \endcode
   * \tparam Offset The offset to the beginning of the slice
   * \tparam Extent The size of the slice
   * \sa views::slice
   */
#ifdef __cpp_lib_ranges
  template<collection T, value::index Offset, value::index Extent> requires
    std::same_as<std::decay_t<Offset>, Offset> and std::same_as<std::decay_t<Extent>, Extent> and
    (not tuple_like<T> or
    ((value::dynamic<Offset> or value::fixed_number_of_v<Offset> <= std::tuple_size_v<std::decay_t<T>>) and
    (value::dynamic<Extent> or value::fixed_number_of_v<Extent> <= std::tuple_size_v<std::decay_t<T>>) and
    (value::dynamic<Offset> or value::dynamic<Extent> or value::fixed_number_of_v<Offset> + value::fixed_number_of_v<Extent> <= std::tuple_size_v<std::decay_t<T>>)))
#else
  template<typename T, typename Offset, typename Extent>
#endif
  struct slice_view : collection_view_interface<slice_view<T, Offset, Extent>>
  {
  private:

    using MyT = std::conditional_t<tuple_like<T> and (value::dynamic<Offset> or value::dynamic<Extent>),
      all_view<T>, OpenKalman::internal::movable_wrapper<T>>;

    constexpr decltype(auto) get_t() & noexcept
    { if constexpr (tuple_like<T> and (value::dynamic<Offset> or value::dynamic<Extent>)) return std::forward<MyT&>(my_t); else return my_t.get(); }

    constexpr decltype(auto) get_t() const & noexcept
    { if constexpr (tuple_like<T> and (value::dynamic<Offset> or value::dynamic<Extent>)) return std::forward<const MyT&>(my_t); else return my_t.get(); }

    constexpr decltype(auto) get_t() && noexcept
    { if constexpr (tuple_like<T> and (value::dynamic<Offset> or value::dynamic<Extent>)) return std::forward<MyT&&>(my_t); else return std::move(*this).my_t.get(); }

    constexpr decltype(auto) get_t() const && noexcept
    { if constexpr (tuple_like<T> and (value::dynamic<Offset> or value::dynamic<Extent>)) return std::forward<const MyT&&>(my_t); else return std::move(*this).my_t.get(); }

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr slice_view() requires std::default_initializable<MyT> and std::default_initializable<Offset> and std::default_initializable<Extent> = default;
#else
    template<typename aT = MyT, std::enable_if_t<std::is_default_constructible_v<aT> and
      std::is_default_constructible_v<Offset> and std::is_default_constructible_v<Extent>, int> = 0>
    constexpr slice_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
#ifdef __cpp_concepts
    template<typename Arg, value::index O, value::index E> requires
      std::constructible_from<MyT, Arg&&> and std::constructible_from<Offset, O&&> and std::constructible_from<Extent, E&&>
#else
    template<typename Arg, typename O, typename E, std::enable_if_t<std::is_constructible_v<MyT, Arg&&> and
      std::is_constructible_v<Offset, O&&> and std::is_constructible_v<Extent, E&&>, int> = 0>
#endif
    explicit constexpr slice_view(Arg&& arg, O o, E e) : my_t {std::forward<Arg>(arg)},
      my_offset{std::move(o)}, my_extent {std::move(e)} {}


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i> requires tuple_like<T>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr (value::fixed<Extent>) static_assert(i < value::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(std::forward<decltype(self)>(self).get_t(),
        value::operation {std::plus{}, std::forward<decltype(self)>(self).my_offset, std::integral_constant<std::size_t, i>{}});
    }
#else
    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() &
    {
      if constexpr (value::fixed<Extent>) static_assert(i < value::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(get_t(),
        value::operation {std::plus{}, my_offset, std::integral_constant<std::size_t, i>{}});
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr (value::fixed<Extent>) static_assert(i < value::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(get_t(),
        value::operation {std::plus{}, my_offset, std::integral_constant<std::size_t, i>{}});
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr (value::fixed<Extent>) static_assert(i < value::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(std::move(*this).get_t(),
        value::operation {std::plus{}, my_offset, std::integral_constant<std::size_t, i>{}});
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr (value::fixed<Extent>) static_assert(i < value::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(std::move(*this).get_t(),
        value::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_collection_size(std::move(*this).get_t())});
    }
#endif


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr value::index auto
    size(this auto&& self) { return std::forward<decltype(self)>(self).my_extent; }
#else
    constexpr auto
    size() const { return my_extent; }
#endif


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    begin(this auto&& self) noexcept requires sized_random_access_range<T> or value::dynamic<Offset> or value::dynamic<Extent>
    {
      return std::ranges::begin(std::forward<decltype(self)>(self).get_t()) + std::forward<decltype(self)>(self).my_offset;
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto begin() & { return ranges::begin(get_t()) + my_offset; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto begin() const & { return ranges::begin(get_t()) + my_offset; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto begin() && noexcept { return ranges::begin(std::move(*this).get_t()) + std::move(*this).my_offset; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto begin() const && noexcept { return ranges::begin(std::move(*this).get_t()) + std::move(*this).my_offset; }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    end(this auto&& self) noexcept requires sized_random_access_range<T> or value::dynamic<Offset> or value::dynamic<Extent>
    {
      return std::ranges::begin(std::forward<decltype(self)>(self).get_t()) +
        std::forward<decltype(self)>(self).my_offset + std::forward<decltype(self)>(self).my_extent;
    }
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto end() & { return ranges::begin(get_t()) + my_offset + my_extent; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto end() const & { return ranges::begin(get_t()) + my_offset + my_extent; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto end() && noexcept { return ranges::begin(std::move(*this).get_t()) + std::move(*this).my_offset + std::move(*this).my_extent; }

    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Offset> or value::dynamic<Extent>, int> = 0>
    constexpr auto end() const && noexcept { return ranges::begin(std::move(*this).get_t()) + std::move(*this).my_offset + std::move(*this).my_extent; }
#endif

  private:

    MyT my_t;

    Offset my_offset;

    Extent my_extent;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<typename Arg, value::index O, value::index E>
#else
  template<typename Arg, typename O, typename E, std::enable_if_t<value::index<O> and value::index<E>, int> = 0>
#endif
  slice_view(Arg&&, const O&, const E&) -> slice_view<Arg, O, E>;

} // namespace OpenKalman


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename T, typename O, typename E>
  constexpr bool enable_borrowed_range<OpenKalman::collections::slice_view<T, O, E>> =
    std::is_lvalue_reference_v<T> or enable_borrowed_range<remove_cvref_t<T>>;
}


namespace std
{
  template<typename T, typename O, typename E>
  struct tuple_size<OpenKalman::collections::slice_view<T, O, E>> : OpenKalman::value::fixed_number_of<E> {};

  template<size_t i, typename T, typename O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<T, O, E>>
      : OpenKalman::collections::internal::maybe_tuple_element<
          OpenKalman::value::operation<std::plus<>, O, std::integral_constant<std::size_t, i>>, std::decay_t<T>> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<value::index O, value::index E>
#else
    template<typename O, typename E>
#endif
    struct slice_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<slice_closure<O, E>>
#endif
    {
      constexpr slice_closure(O o, E e) : offset {std::move(o)}, extent {std::move(e)} {};

#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const { return slice_view {std::forward<R>(r), offset, extent}; }

    private:
      O offset;
      E extent;
    };


    struct slice_adapter
    {
#ifdef __cpp_concepts
      template<value::index O, value::index E>
#else
      template<typename O, typename E, std::enable_if_t<value::index<O> and value::index<E>, int> = 0>
#endif
      constexpr auto
      operator() (O o, E e) const
      {
        return slice_closure<O, E> {std::move(o), std::move(e)};
      }


#ifdef __cpp_concepts
      template<collection T, value::index O, value::index E>
#else
      template<typename T, typename O, typename E, std::enable_if_t<collection<T> and value::index<O> and value::index<E>, int> = 0>
#endif
      constexpr auto
      operator() (T&& t, O o, E e) const
      {
        return slice_view {std::forward<T>(t), std::move(o), std::move(e)};
      }

    };

  }


  /**
   * \brief a RangeAdapterObject associated with \ref slice_view.
   * \details The expression <code>views::slice(arg)</code> is expression-equivalent
   * to <code>slice_view(arg)</code> for any suitable \ref collection arg.
   * \sa slice_view
   */
  inline constexpr detail::slice_adapter slice;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
