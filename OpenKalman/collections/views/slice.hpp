/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "values/values.hpp"
#include "collections/functions/get_size.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get_element.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view representing a slice of a \ref collection.
   * \tparam Offset The offset to the beginning of the slice
   * \tparam Extent The size of the slice. If this is omitted or \ref values::unbounded_size_t,
   * The slice will run to the end of the collection (if it is bounded).
   * \sa views::slice
   */
#ifdef __cpp_lib_ranges
  template<collection V, values::index Offset, values::size Extent = values::unbounded_size_t> requires
    std::same_as<std::decay_t<Offset>, Offset> and std::same_as<std::decay_t<Extent>, Extent> and
    (not values::size_compares_with<Offset, size_of_t<V>, &stdex::is_gt>) and
    (not values::index<Extent> or not values::size_compares_with<Extent, size_of_t<V>, &stdex::is_gt>) and
    (not sized<V> or not values::index<Extent> or
      not values::size_compares_with<values::operation_t<std::plus<>, Offset, Extent>, size_of_t<V>, &std::is_gt>)
#else
  template<typename V, typename Offset, typename Extent = values::unbounded_size_t>
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
    slice_view(V&& v, Offset offset = {}, Extent extent = {})
      : v_ {std::forward<V>(v)}, offset_ {std::move(offset)}, extent_ {std::move(extent)} {}


    /**
     * \brief The base view.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).v_.get(); }
#else
    constexpr V& base() & { return this->v_.get(); }
    constexpr const V& base() const & { return this->v_.get(); }
    constexpr V&& base() && noexcept { return std::move(*this).v_.get(); }
    constexpr const V&& base() const && noexcept { return std::move(*this).v_.get(); }
#endif

#ifndef __cpp_explicit_this_parameter
  private:

    template<typename Self>
    static constexpr auto
    begin_impl(Self&& self) noexcept
    {
      return stdex::ranges::begin(std::forward<Self>(self).v_.get());
    }

    template<typename Self>
    static constexpr auto
    end_impl(Self&& self) noexcept
    {
      return stdex::ranges::end(std::forward<Self>(self).v_.get());
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
      return stdex::ranges::begin(std::forward<decltype(self)>(self).v_.get()) + std::forward<decltype(self)>(self).offset_;
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
      if constexpr (values::index<Extent>)
        return std::forward<decltype(self)>(self).begin() + std::forward<decltype(self)>(self).extent_;
      else
        return stdex::ranges::end(std::forward<decltype(self)>(self).v_.get());
    }
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() &
    {
      if constexpr (values::index<Extent>) return this->begin() + extent_;
      else return end_impl(*this);
    }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() const &
    {
      if constexpr (values::index<Extent>) return this->begin() + extent_;
      else return end_impl(*this);
    }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() && noexcept
    {
      if constexpr (values::index<Extent>) return std::move(*this).begin() + std::move(*this).extent_;
      else return end_impl(std::move(*this));
    }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::range<const V>, int> = 0>
    constexpr auto end() const && noexcept
    {
      if constexpr (values::index<Extent>) return std::move(*this).begin() + std::move(*this).extent_;
      else return end_impl(std::move(*this));
    }
#endif


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr values::size auto
    size(this auto&& self) noexcept requires values::index<Extent> or sized<V>
    {
      if constexpr (values::index<Extent>)
        return std::forward<decltype(self)>(self).extent_;
      else
        return values::operation(
          std::minus{},
          get_size(std::forward<decltype(self)>(self).v_.get()),
          std::forward<decltype(self)>(self).offset_);
    }
#else
    template<bool Enable = true, std::enable_if_t<values::index<Extent> or sized<V>, int> = 0>
    constexpr auto
    size() const noexcept
    {
      if constexpr (values::index<Extent>)
        return extent_;
      else
        return values::operation(std::minus{}, get_size(v_.get()), offset_);
    }
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
      return collections::get_element(std::forward<decltype(self)>(self).v_.get(),
        values::operation(std::plus{}, std::forward<decltype(self)>(self).offset_, std::integral_constant<std::size_t, i>{}));
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(v_.get(),
        values::operation(std::plus{}, offset_, std::integral_constant<std::size_t, i>{}));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(v_.get(),
        values::operation(std::plus{}, offset_, std::integral_constant<std::size_t, i>{}));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(std::move(*this).v_.get(),
        values::operation(std::plus{}, offset_, std::integral_constant<std::size_t, i>{}));
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_value_of_v<Extent>, "Index exceeds range");
      return collections::get_element(std::move(*this).v_.get(),
        values::operation(std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(std::move(*this).get_t())));
    }
#endif

  private:

    internal::movable_wrapper<V> v_;
    Offset offset_;
    Extent extent_;

  };


  /**
   * \brief Deduction guides
   */
  template<typename V, typename O, typename E>
  slice_view(V&&, const O&, const E&) -> slice_view<V, O, E>;

  template<typename V, typename O>
  slice_view(V&&, const O&) -> slice_view<V, O>;


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

  template<typename V, typename O>
  struct tuple_size<OpenKalman::collections::slice_view<V, O, OpenKalman::values::unbounded_size_t>>
    : std::conditional_t<
        OpenKalman::values::fixed<OpenKalman::collections::size_of<V>> and OpenKalman::values::fixed<O>,
        OpenKalman::values::operation_t<std::minus<>, OpenKalman::collections::size_of<V>, O>,
        std::monostate> {};


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
    template<typename R, typename O, typename E>
    constexpr auto
    slice_impl(R&& r, O&& o, E&& e)
    {
      if constexpr (viewable_collection<R>)
      {
        if constexpr (values::fixed_value_compares_with<O, 0_uz> and values::size_compares_with<E, size_of_t<R>>)
          return all(std::forward<R>(r));
        else
          return slice_view {all(std::forward<R>(r)), std::forward<O>(o), std::forward<E>(e)};
      }
      else return slice_view {std::forward<R>(r), std::forward<O>(o), std::forward<E>(e)};
    };


    /**
     * \internal
     * \brief A closure for the slice view
     */
    template<typename O, typename E>
    struct slice_closure : stdex::ranges::range_adaptor_closure<slice_closure<O, E>>
    {
      constexpr slice_closure(O o, E e) : offset_ {std::move(o)}, extent_ {std::move(e)} {};

#ifdef __cpp_concepts
      template<typename R> requires
        (viewable_collection<R> or
          (uniformly_gettable<R> and values::fixed<O> and (values::fixed<E> or not values::index<E>))) and
        (not values::size_compares_with<O, size_of<R>, &std::is_gt>) and
        (not values::index<E> or not values::size_compares_with<E, size_of<R>, &std::is_gt>) and
        (not sized<R> or size_of_v<R> == std::dynamic_extent or
          values::dynamic<O> or values::dynamic<E> or
          values::fixed_value_of_v<O> + values::fixed_value_of_v<E> <= size_of_v<R>)
#else
      template<typename R, std::enable_if_t<
        ((viewable_collection<R> or (uniformly_gettable<R> and values::fixed<O> and (values::fixed<E> or not values::index<E>))) and
          values::index<O> and values::size<E>) and
        (not values::size_compares_with<O, size_of<R>, &stdex::is_gt>) and
        (not values::index<E> or not values::size_compares_with<E, size_of<R>, &stdex::is_gt>), int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const
      {
        return slice_impl(std::forward<R>(r), offset_, extent_);
      }

    private:
      O offset_;
      E extent_;
    };


    struct slice_adapter
    {
#ifdef __cpp_concepts
      template<values::index O, values::size E = values::unbounded_size_t>
#else
      template<typename O, typename E = values::unbounded_size_t, std::enable_if_t<
        values::index<O> and values::size<E>, int> = 0>
#endif
      constexpr auto
      operator() (O o, E e = values::unbounded_size) const
      {
        return slice_closure<O, E> {std::move(o), std::move(e)};
      }


#ifdef __cpp_concepts
      template<typename R, values::index O, values::size E = values::unbounded_size_t> requires
        (viewable_collection<R> or
          (uniformly_gettable<R> and values::fixed<O> and (values::fixed<E> or not values::index<E>))) and
        (not values::size_compares_with<O, size_of_t<R>, &std::is_gt>) and
        (not values::index<E> or not values::size_compares_with<E, size_of_t<R>, &std::is_gt>) and
        (not sized<R> or not values::index<E> or
          not values::size_compares_with<values::operation_t<std::plus<>, O, E>, size_of_t<R>, &std::is_gt>)
#else
      template<typename R, typename O, typename E = values::unbounded_size_t, std::enable_if_t<
        ((viewable_collection<R> or (uniformly_gettable<R> and values::fixed<O> and (values::fixed<E> or not values::index<E>))) and
          values::index<O> and values::size<E>) and
        (not values::size_compares_with<O, size_of<R>, &stdex::is_gt>) and
        (not values::index<E> or not values::size_compares_with<E, size_of<R>, &stdex::is_gt>), int> = 0>
#endif
      constexpr decltype(auto)
      operator() (R&& r, O o, E e = values::unbounded_size) const
      {
        return slice_impl(std::forward<R>(r), std::move(o), std::move(e));
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
