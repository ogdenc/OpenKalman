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
 * \brief Definition of \ref collections::slice_view and \ref collections::views::slice.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP

#include <tuple>
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/functions/operation.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get.hpp"
#include "collections/functions/lexicographical_compare_three_way.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
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
  template<collection V, values::index Offset, values::index Extent> requires
    std::same_as<std::decay_t<Offset>, Offset> and std::same_as<std::decay_t<Extent>, Extent> and
    (not sized<V> or size_of_v<V> == stdex::dynamic_extent or
    ((values::dynamic<Offset> or values::fixed_value_of_v<Offset> <= size_of_v<V>) and
    (values::dynamic<Extent> or values::fixed_value_of_v<Extent> <= size_of_v<V>) and
    (values::dynamic<Offset> or values::dynamic<Extent> or values::fixed_value_of_v<Offset> + values::fixed_value_of_v<Extent> <= size_of_v<V>)))
#else
  template<typename V, typename Offset, typename Extent>
#endif
  struct slice_view : stdex::ranges::view_interface<slice_view<V, Offset, Extent>>
  {
    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr slice_view() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<V> and
      stdex::default_initializable<Offset> and stdex::default_initializable<Extent>, int> = 0>
    constexpr slice_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
    constexpr
    slice_view(const V& v, Offset offset, Extent extent)
      : v_ {v}, offset_ {std::move(offset)}, extent_ {std::move(extent)} {}

    /// \overload
    constexpr
    slice_view(V&& v, Offset offset, Extent extent)
      : v_ {std::move(v)}, offset_ {std::move(offset)}, extent_ {std::move(extent)} {}


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

#ifndef __cpp_explicit_this_parameter
  private:

    template<typename Self>
    static constexpr auto
    begin_impl(Self&& self) noexcept
    {
      return stdex::ranges::begin(std::forward<Self>(self).v_);
    }

  public:
#endif

    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    begin(this auto&& self) noexcept requires stdex::ranges::range<const V>
    {
      return stdex::ranges::begin(std::forward<decltype(self)>(self).v_) + std::forward<decltype(self)>(self).offset_;
    }
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto begin() & { return begin_impl(*this) + offset_; }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto begin() const & { return begin_impl(*this) + offset_; }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto begin() && noexcept { return begin_impl(std::move(*this)) + std::move(*this).offset_; }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto begin() const && noexcept { return begin_impl(std::move(*this)) + std::move(*this).offset_; }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    end(this auto&& self) noexcept requires stdex::ranges::range<const V>
    {
      return std::forward<decltype(self)>(self).begin() + std::forward<decltype(self)>(self).extent_;
    }
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() & { return this->begin() + extent_; }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() const & { return this->begin() + extent_; }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() && noexcept { return std::move(*this).begin() + std::move(*this).extent_; }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() const && noexcept { return std::move(*this).begin() + std::move(*this).extent_; }
#endif


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr values::index auto
    size(this auto&& self) noexcept { return std::forward<decltype(self)>(self).extent_; }
#else
    constexpr auto
    size() const noexcept { return extent_; }
#endif


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(std::forward<decltype(self)>(self).v_,
        values::operation(std::plus{}, std::forward<decltype(self)>(self).offset_, std::integral_constant<std::size_t, i>{}));
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(v_,
        values::operation(std::plus{}, offset_, std::integral_constant<std::size_t, i>{}));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(v_,
        values::operation(std::plus{}, offset_, std::integral_constant<std::size_t, i>{}));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(std::move(*this).v_,
        values::operation(std::plus{}, offset_, std::integral_constant<std::size_t, i>{}));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(std::move(*this).v_,
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(std::move(*this).get_t())));
    }
#endif

  private:

    V v_;
    Offset offset_;
    Extent extent_;

  };


  /**
   * \brief Deduction guide
   */
  template<typename V, typename O, typename E>
  slice_view(const V&, const O&, const E&) -> slice_view<V, O, E>;

}


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdex::ranges
#endif
{
  template<typename V, typename O, typename E>
  constexpr bool enable_borrowed_range<OpenKalman::collections::slice_view<V, O, E>> = enable_borrowed_range<V>;
}


#ifndef __cpp_lib_ranges
namespace OpenKalman::collections::detail
{
  template<std::size_t i, typename V, typename O, typename = void>
  struct slice_tuple_element {};

  template<std::size_t i, typename V, typename O>
  struct slice_tuple_element<i, V, O, std::enable_if_t<values::fixed<O>>>
    : collection_element<values::fixed_value_of_v<O> + i, std::decay_t<V>> {};
}
#endif


namespace std
{
  template<typename V, typename O, typename E>
  struct tuple_size<OpenKalman::collections::slice_view<V, O, E>> : OpenKalman::values::fixed_value_of<E> {};


#ifdef __cpp_concepts
  template<size_t i, typename V, typename O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<V, O, E>> {};

  template<size_t i, typename V, OpenKalman::values::fixed O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<V, O, E>>
    : OpenKalman::collections::collection_element<OpenKalman::values::fixed_value_of_v<O> + i, std::decay_t<V>> {};
#else
  template<size_t i, typename V, typename O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<V, O, E>>
    : OpenKalman::collections::detail::slice_tuple_element<i, V, O> {};
#endif
}


namespace OpenKalman::collections::views
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename O, typename E, typename N>
#else
    template<typename O, typename E, typename N, typename = void>
#endif
    struct slice_is_whole : std::false_type {};

#ifdef __cpp_concepts
    template<values::fixed O, values::fixed E, values::fixed N>
    struct slice_is_whole<O, E, N>
#else
    template<typename O, typename E, typename N>
    struct slice_is_whole<O, E, N, std::enable_if_t<values::fixed<O> and values::fixed<E> and values::fixed<N>>>
#endif
      : std::bool_constant<(values::fixed_value_of_v<O> == 0_uz and values::fixed_value_of_v<E> == values::fixed_value_of_v<N>)> {};



    template<typename O, typename E>
    struct slice_closure : stdex::ranges::range_adaptor_closure<slice_closure<O, E>>
    {
      constexpr slice_closure(O o, E e) : offset_ {std::move(o)}, extent_ {std::move(e)} {};

#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const
      {
        if constexpr (slice_is_whole<O, E, size_of<R>>::value)
          return all(std::forward<R>(r));
        else
          return slice_view {all(std::forward<R>(r)), offset_, extent_};
      }

    private:
      O offset_;
      E extent_;
    };


    struct slice_adapter
    {
#ifdef __cpp_concepts
      template<values::index O, values::index E>
#else
      template<typename O, typename E, std::enable_if_t<values::index<O> and values::index<E>, int> = 0>
#endif
      constexpr auto
      operator() (O o, E e) const
      {
        return slice_closure<O, E> {std::move(o), std::move(e)};
      }


#ifdef __cpp_concepts
      template<viewable_collection R, values::index O, values::index E>
#else
      template<typename R, typename O, typename E, std::enable_if_t<
        viewable_collection<R> and values::index<O> and values::index<E>, int> = 0>
#endif
      constexpr decltype(auto)
      operator() (R&& r, O o, E e) const
      {
        if constexpr (slice_is_whole<O, E, size_of<R>>::value)
          return all(std::forward<R>(r));
        else
          return slice_view {all(std::forward<R>(r)), std::move(o), std::move(e)};
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


#endif
